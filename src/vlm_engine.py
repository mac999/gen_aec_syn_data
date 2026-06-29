"""
VLM synthesis engine — ComfyUI headless API integration.

Workflow
--------
1. Upload a BIM render PNG to the ComfyUI `/upload/image` endpoint.
2. Submit a ControlNet-Canny + SD-1.5 img2img workflow via `/prompt`.
3. Poll `/history/{prompt_id}` until the job finishes.
4. Download the synthetic site-photo output via `/view`.
5. Pair (bim_render, site_photo, IFC metadata) into a VLMSample and write JSONL.

If ComfyUI is unreachable the engine falls back to copying the BIM render as
the "site photo" so the pipeline can still produce structurally-valid VLM
records for manual review.
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import requests
from .config import PipelineConfig
from .schemas import IFCElementInfo, VLMMetadata, VLMOutput, VLMSample

logger = logging.getLogger("AEC_Pipeline.vlm_engine")

# ComfyUI workflow template (SD1.5 + ControlNet Canny img2img) 
def _build_comfyui_workflow(
    uploaded_image_name: str,
    sd_model: str,
    controlnet_model: str,
    denoise: float,
    steps: int,
    cfg: float,
    controlnet_strength: float,
    seed: int,
    use_canny_preprocessor: bool = True,
) -> Dict[str, Any]:
    """
    Return a ComfyUI API-format workflow dict.

    When *use_canny_preprocessor* is False (CannyEdgePreprocessor node not
    installed), the raw loaded image is fed directly into ControlNetApply
    instead of going through edge detection first.
    """
    # Node "2" is either a CannyEdgePreprocessor or a passthrough identity.
    # The ControlNetApply node always reads from node "2" output index 0.
    if use_canny_preprocessor:
        node_2: Dict[str, Any] = {
            "class_type": "CannyEdgePreprocessor",
            "inputs": {
                "image": ["1", 0],
                "low_threshold": 100,
                "high_threshold": 200,
                "resolution": 512,
            },
        }
    else:
        # ImageScale acts as a passthrough at 1:1 ratio — universally available
        node_2 = {
            "class_type": "ImageScale",
            "inputs": {
                "image": ["1", 0],
                "upscale_method": "nearest-exact",
                "width": 512,
                "height": 512,
                "crop": "disabled",
            },
        }

    return {
        "1": {
            "class_type": "LoadImage",
            "inputs": {"image": uploaded_image_name},
        },
        "2": node_2,
        "3": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": sd_model},
        },
        "4": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "clip": ["3", 1],
                "text": (
                    "construction site, realistic photograph, "
                    "high detail, professional photography"
                ),
            },
        },
        "5": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "clip": ["3", 1],
                "text": (
                    "blurry, cartoon, drawing, illustration, "
                    "low quality, watermark, text, logo, sky, clouds, people, animals"
                ),
            },
        },
        "6": {
            "class_type": "ControlNetLoader",
            "inputs": {"control_net_name": controlnet_model},
        },
        "7": {
            "class_type": "ControlNetApply",
            "inputs": {
                "conditioning": ["4", 0],
                "control_net": ["6", 0],
                "image": ["2", 0],
                "strength": controlnet_strength,
            },
        },
        "8": {
            "class_type": "VAEEncode",
            "inputs": {
                "pixels": ["1", 0],
                "vae": ["3", 2],
            },
        },
        "9": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["3", 0],
                "positive": ["7", 0],
                "negative": ["5", 0],
                "latent_image": ["8", 0],
                "seed": seed,
                "steps": steps,
                "cfg": cfg,
                "sampler_name": "euler_ancestral",
                "scheduler": "normal",
                "denoise": denoise,
            },
        },
        "10": {
            "class_type": "VAEDecode",
            "inputs": {
                "samples": ["9", 0],
                "vae": ["3", 2],
            },
        },
        "11": {
            "class_type": "SaveImage",
            "inputs": {
                "images": ["10", 0],
                "filename_prefix": "site_photo",
            },
        },
    }


class VLMEngine:
    """
    Synthesises VLM training samples from IFC renders via ComfyUI.
    Falls back gracefully when ComfyUI is not running.
    """

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self._comfyui_available: Optional[bool] = None
        self._resolved_controlnet: Optional[str] = None
        self._resolved_checkpoint: Optional[str] = None
        self._canny_node_available: Optional[bool] = None
        self._sample_counter = 0

        self.config.vlm_output_dir.mkdir(parents=True, exist_ok=True)
        self.config.site_photo_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.config.vlm_output_dir / "vlm_training_data.jsonl"

    def process_renders(
        self,
        render_paths: List[Path],
        elements: List[IFCElementInfo],
        model_id: str,
        project_type: str = "건물",
        trade_type: str = "철근콘크리트",
    ) -> int:
        """
        For each BIM render, synthesise a site photo and write a VLMSample.

        Returns the number of successfully written samples.
        """
        successful = 0
        elem_ids = [e.global_id for e in elements[:10]]  # cap to 10

        for render_path in render_paths:
            view_name = render_path.stem.split("_")[-1]
            site_photo_path = self._synthesise_site_photo(render_path)

            if site_photo_path is None:
                logger.warning("Skipping render %s — site photo synthesis failed", render_path.name)
                continue

            sample = self._build_vlm_sample(
                render_path=render_path,
                site_photo_path=site_photo_path,
                elem_ids=elem_ids,
                project_type=project_type,
                trade_type=trade_type,
                view_type=f"3d_{view_name}" if view_name != "top" else "top_view",
            )
            self._append_sample(sample)
            successful += 1

        return successful

    def _synthesise_site_photo(self, render_path: Path) -> Optional[Path]:
        """
        Try ComfyUI synthesis; fall back to a direct copy of the BIM render.
        """
        if self._is_comfyui_available():
            result = self._run_comfyui(render_path)
            if result:
                return result
            logger.warning("ComfyUI synthesis failed — using BIM render as fallback.")

        # Fallback: copy the BIM render as the site photo
        return self._copy_as_site_photo(render_path)

    def _is_comfyui_available(self) -> bool:
        if self._comfyui_available is not None:
            return self._comfyui_available
        try:
            resp = requests.get(
                f"{self.config.comfyui_url}/system_stats", timeout=3
            )
            self._comfyui_available = resp.status_code == 200
        except Exception:
            self._comfyui_available = False

        if self._comfyui_available:
            logger.info("ComfyUI detected at %s", self.config.comfyui_url)
        else:
            logger.warning(
                "ComfyUI not reachable at %s — VLM images will use BIM renders.",
                self.config.comfyui_url,
            )
        return self._comfyui_available

    def _fetch_object_info(self, class_type: str) -> Dict[str, Any]:
        """Query /object_info/<class_type> and return the parsed JSON."""
        url = f"{self.config.comfyui_url}/object_info/{class_type}"
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        return resp.json()

    def _resolve_models(self) -> bool:
        """
        Query ComfyUI for installed ControlNet and checkpoint models.
        Selects the best available match and caches the result.
        Returns False if no usable models are found.
        """
        if self._resolved_controlnet and self._resolved_checkpoint:
            return True

        # ── ControlNet ────────────────────────────────────────────────────
        try:
            info = self._fetch_object_info("ControlNetLoader")
            available: List[str] = (
                info.get("ControlNetLoader", {})
                .get("input", {})
                .get("required", {})
                .get("control_net_name", [[]])[0]
            )
        except Exception as exc:
            logger.error("Failed to query ControlNetLoader object_info: %s", exc)
            available = []

        if not available:
            logger.error(
                "No ControlNet models found in ComfyUI (models/controlnet/ is empty).\n"
                "  Download a model — e.g.:\n"
                "    https://huggingface.co/lllyasviel/ControlNet-v1-1\n"
                "  and place the .pth/.safetensors file in ComfyUI/models/controlnet/"
            )
            return False

        # Prefer canny-related model matching the configured name; fall back
        # to first canny model found; then first model of any kind.
        preferred = self.config.controlnet_model.lower()
        canny_models = [m for m in available if "canny" in m.lower()]
        if preferred in [m.lower() for m in available]:
            selected_cn = next(m for m in available if m.lower() == preferred)
        elif canny_models:
            selected_cn = canny_models[0]
        else:
            selected_cn = available[0]

        if selected_cn != self.config.controlnet_model:
            logger.warning(
                "Configured ControlNet model '%s' not found. "
                "Auto-selected '%s' from %d available model(s): %s",
                self.config.controlnet_model,
                selected_cn,
                len(available),
                available,
            )
        else:
            logger.info("ControlNet model resolved: %s", selected_cn)
        self._resolved_controlnet = selected_cn

        # ── SD Checkpoint ─────────────────────────────────────────────────
        try:
            info = self._fetch_object_info("CheckpointLoaderSimple")
            ckpts: List[str] = (
                info.get("CheckpointLoaderSimple", {})
                .get("input", {})
                .get("required", {})
                .get("ckpt_name", [[]])[0]
            )
        except Exception as exc:
            logger.error("Failed to query CheckpointLoaderSimple object_info: %s", exc)
            ckpts = []

        if not ckpts:
            logger.error(
                "No checkpoint models found in ComfyUI (models/checkpoints/ is empty).\n"
                "  Download SD 1.5 from:\n"
                "    https://huggingface.co/runwayml/stable-diffusion-v1-5"
            )
            return False

        preferred_ckpt = self.config.sd_base_model.lower()
        if preferred_ckpt in [c.lower() for c in ckpts]:
            selected_ckpt = next(c for c in ckpts if c.lower() == preferred_ckpt)
        else:
            selected_ckpt = ckpts[0]
            logger.warning(
                "Configured checkpoint '%s' not found. "
                "Auto-selected '%s' from %d available: %s",
                self.config.sd_base_model,
                selected_ckpt,
                len(ckpts),
                ckpts,
            )
        self._resolved_checkpoint = selected_ckpt

        # ── CannyEdgePreprocessor availability ────────────────────────────
        try:
            info = self._fetch_object_info("CannyEdgePreprocessor")
            self._canny_node_available = "CannyEdgePreprocessor" in info
        except Exception:
            self._canny_node_available = False

        if not self._canny_node_available:
            logger.warning(
                "CannyEdgePreprocessor node not found — the raw BIM render will be "
                "passed directly to ControlNet. Install comfyui_controlnet_aux for "
                "better results: https://github.com/Fannovel16/comfyui_controlnet_aux"
            )

        return True

    def _run_comfyui(self, render_path: Path) -> Optional[Path]:
        """Full ComfyUI pipeline: upload → queue → poll → download."""
        try:
            if not self._resolve_models():
                logger.error(
                    "Aborting ComfyUI run — required models are not installed. "
                    "See above for installation instructions."
                )
                return None

            uploaded_name = self._upload_image(render_path)
            if not uploaded_name:
                return None

            seed = 145881275571499 # int(uuid.uuid4().int & 0xFFFFFFFF)
            workflow = _build_comfyui_workflow(
                uploaded_image_name=uploaded_name,
                sd_model=self._resolved_checkpoint,
                controlnet_model=self._resolved_controlnet,
                denoise=self.config.i2i_denoise,
                steps=self.config.i2i_steps,
                cfg=self.config.i2i_cfg,
                controlnet_strength=self.config.controlnet_strength,
                seed=seed,
                use_canny_preprocessor=self._canny_node_available,
            )

            client_id = str(uuid.uuid4())
            prompt_id = self._queue_prompt(workflow, client_id)
            if not prompt_id:
                return None

            output_image_name = self._poll_until_done(prompt_id)
            if not output_image_name:
                return None

            return self._download_image(output_image_name, render_path.stem)

        except Exception as exc:
            logger.error("ComfyUI run error: %s", exc)
            return None

    def _upload_image(self, image_path: Path) -> Optional[str]:
        url = f"{self.config.comfyui_url}/upload/image"
        with open(image_path, "rb") as fh:
            files = {"image": (image_path.name, fh, "image/png")}
            resp = requests.post(url, files=files, timeout=30)
        if resp.status_code == 200:
            return resp.json().get("name")
        logger.error("Image upload failed: %s %s", resp.status_code, resp.text)
        return None

    def _queue_prompt(
        self, workflow: Dict[str, Any], client_id: str
    ) -> Optional[str]:
        url = f"{self.config.comfyui_url}/prompt"
        payload = {"prompt": workflow, "client_id": client_id}
        resp = requests.post(url, json=payload, timeout=30)
        if resp.status_code == 200:
            return resp.json().get("prompt_id")
        logger.error("Prompt queue failed: %s %s", resp.status_code, resp.text)
        return None

    def _poll_until_done(self, prompt_id: str) -> Optional[str]:
        url = f"{self.config.comfyui_url}/history/{prompt_id}"
        deadline = time.time() + self.config.comfyui_timeout

        while time.time() < deadline:
            try:
                resp = requests.get(url, timeout=10)
                if resp.status_code == 200:
                    history = resp.json()
                    if prompt_id in history:
                        outputs = history[prompt_id].get("outputs", {})
                        for node_id, node_output in outputs.items():
                            images = node_output.get("images", [])
                            if images:
                                return images[0].get("filename")
            except Exception as exc:
                logger.debug("Polling error: %s", exc)
            time.sleep(5)

        logger.error("ComfyUI job timed out after %ds", self.config.comfyui_timeout)
        return None

    def _download_image(
        self, image_name: str, stem: str
    ) -> Optional[Path]:
        url = f"{self.config.comfyui_url}/view"
        params = {"filename": image_name, "type": "output"}
        resp = requests.get(url, params=params, timeout=30)
        if resp.status_code != 200:
            logger.error("Image download failed: %s", resp.status_code)
            return None

        out_path = self.config.site_photo_dir / f"{stem}_site.jpg"
        out_path.write_bytes(resp.content)
        logger.info("Site photo saved: %s", out_path.name)
        return out_path

    def _copy_as_site_photo(self, render_path: Path) -> Path:
        """Fallback — copy the BIM render as site photo placeholder."""
        import shutil  # noqa: PLC0415
        out_path = self.config.site_photo_dir / f"{render_path.stem}_site.png"
        shutil.copy2(render_path, out_path)
        logger.info("Fallback site photo saved (copy): %s", out_path.name)
        return out_path

    def _build_vlm_sample(
        self,
        render_path: Path,
        site_photo_path: Path,
        elem_ids: List[str],
        project_type: str,
        trade_type: str,
        view_type: str,
    ) -> VLMSample:
        self._sample_counter += 1
        sample_id = f"vlm_{self._sample_counter:06d}"

        # Relative paths (relative to output root for portability)
        bim_rel = str(render_path.relative_to(self.config.output_dir)).replace("\\", "/")
        site_rel = str(site_photo_path.relative_to(self.config.output_dir)).replace("\\", "/")

        return VLMSample(
            id=sample_id,
            task_type="bim_site_alignment",
            images=[f"images/bim_render/{render_path.name}", f"images/site_photo/{site_photo_path.name}"],
            metadata=VLMMetadata(
                project_type=project_type,
                bim_element_ids=elem_ids,
                trade_type=trade_type,
                view_type=view_type,
            ),
            instruction=(
                "현장 사진이 BIM 설계 상태와 일치하는지 판단하고 "
                "불일치 요소와 그 근거를 구체적으로 설명하라."
            ),
            output=VLMOutput(
                answer=(
                    "BIM 렌더링과 현장 사진의 일치 여부를 분석한 결과, "
                    "주요 구조 요소는 설계와 일치하나 세부 마감 작업이 "
                    "미완료 상태로 확인됨."
                ),
                label="partial_match",
                evidence=[
                    "BIM 모델의 구조 배치와 현장 사진의 구조물 위치가 일치함",
                    "현장 사진에서 일부 마감재 및 설치물이 설계 대비 미설치 상태로 확인됨",
                ],
            ),
        )

    def _append_sample(self, sample: VLMSample) -> None:
        with open(self.jsonl_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(sample.to_jsonl_dict(), ensure_ascii=False) + "\n")
        logger.info("Appended VLM sample %s", sample.id)
