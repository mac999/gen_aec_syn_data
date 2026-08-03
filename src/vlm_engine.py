"""
VLM synthesis engine — ComfyUI headless API integration.

Workflow
--------
1. Upload the ControlNet hint (a depth map rasterised from the IFC mesh) to
   the ComfyUI `/upload/image` endpoint.
2. Submit a ControlNet + SD-1.5 workflow via `/prompt`.
3. Poll `/history/{prompt_id}` until the job finishes.
4. Download the synthetic site-photo output via `/view`.
5. Pair (bim_render, site_photo, IFC metadata) into a VLMSample and write JSONL.

Why the geometry is carried by ControlNet and not by img2img
-----------------------------------------------------------
Starting the sampler from the BIM render (img2img at denoise < 1) leaves part
of the render's flat shading in the final latent, so outputs came back as
recoloured wireframes rather than photographs. The BIM form is therefore held
by a ControlNet depth hint while the latent starts empty, which lets the prompt
decide every surface. Set ``vlm_init_from_render`` to restore the old
behaviour.

If ComfyUI is unreachable the engine falls back to copying the BIM render as
the "site photo" so the pipeline can still produce structurally-valid VLM
records for manual review.
"""
from __future__ import annotations

import base64
import json
import logging
import os
import time
import uuid
import zlib
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import requests
from .config import PipelineConfig
from .schemas import IFCElementInfo, VLMMetadata, VLMOutput, VLMSample
from .sllm_sft_engine import _extract_json_object  # reuse robust JSON recovery

logger = logging.getLogger("AEC_Pipeline.vlm_engine")

def _fmt_prompt(template: str, **fields: str) -> str:
    """Substitute {project_type}/{trade_type}/{view_type} without dying on typos."""
    try:
        return template.format(**fields)
    except (KeyError, IndexError, ValueError) as exc:
        logger.warning(
            "Prompt template contains an unknown placeholder (%s) — "
            "using it verbatim. Valid names: %s",
            exc, ", ".join(sorted(fields)),
        )
        return template


def _build_comfyui_workflow(
    control_image_name: str,
    sd_model: str,
    controlnet_model: str,
    positive: str,
    negative: str,
    width: int,
    height: int,
    steps: int,
    cfg: float,
    sampler: str,
    scheduler: str,
    denoise: float,
    controlnet_strength: float,
    seed: int,
    control_start: float = 0.0,
    control_end: float = 1.0,
    init_image_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Return a ComfyUI API-format workflow dict.

    The ControlNet hint (node "1") is scaled to the generation size and drives
    the structure. When *init_image_name* is given the sampler starts from that
    image instead of an empty latent, which only makes sense with
    ``denoise`` < 1.
    """
    workflow: Dict[str, Any] = {
        "1": {
            "class_type": "LoadImage",
            "inputs": {"image": control_image_name},
        },
        # Match the hint to the latent resolution; a mismatched hint gets
        # stretched internally and blurs the structure it is meant to pin.
        "2": {
            "class_type": "ImageScale",
            "inputs": {
                "image": ["1", 0],
                "upscale_method": "bilinear",
                "width": width,
                "height": height,
                "crop": "disabled",
            },
        },
        "3": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": sd_model},
        },
        "4": {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": ["3", 1], "text": positive},
        },
        "5": {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": ["3", 1], "text": negative},
        },
        "6": {
            "class_type": "ControlNetLoader",
            "inputs": {"control_net_name": controlnet_model},
        },
        # Advanced apply so the hint can be released before the last steps —
        # holding it to 100% keeps surfaces looking like shaded geometry.
        "7": {
            "class_type": "ControlNetApplyAdvanced",
            "inputs": {
                "positive": ["4", 0],
                "negative": ["5", 0],
                "control_net": ["6", 0],
                "image": ["2", 0],
                "strength": controlnet_strength,
                "start_percent": control_start,
                "end_percent": control_end,
            },
        },
        "9": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["3", 0],
                "positive": ["7", 0],
                "negative": ["7", 1],
                "latent_image": ["8", 0],
                "seed": seed,
                "steps": steps,
                "cfg": cfg,
                "sampler_name": sampler,
                "scheduler": scheduler,
                "denoise": denoise,
            },
        },
        "10": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["9", 0], "vae": ["3", 2]},
        },
        "11": {
            "class_type": "SaveImage",
            "inputs": {"images": ["10", 0], "filename_prefix": "site_photo"},
        },
    }

    if init_image_name:
        workflow["12"] = {
            "class_type": "LoadImage",
            "inputs": {"image": init_image_name},
        }
        workflow["13"] = {
            "class_type": "ImageScale",
            "inputs": {
                "image": ["12", 0],
                "upscale_method": "bilinear",
                "width": width,
                "height": height,
                "crop": "disabled",
            },
        }
        workflow["8"] = {
            "class_type": "VAEEncode",
            "inputs": {"pixels": ["13", 0], "vae": ["3", 2]},
        }
    else:
        workflow["8"] = {
            "class_type": "EmptyLatentImage",
            "inputs": {"width": width, "height": height, "batch_size": 1},
        }

    return workflow


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
        self._sample_counter = 0

        # Output locations are set per input file via set_output_dir(); the
        # defaults below keep the engine usable standalone.
        self.output_root = self.config.vlm_output_dir
        self.bim_render_dir = self.config.bim_render_dir
        self.site_photo_dir = self.config.site_photo_dir
        self.depth_map_dir = self.config.vlm_output_dir / "images" / "depth"
        self.jsonl_path = self.config.vlm_output_dir / "vlm_training_data.jsonl"

    def set_output_dir(self, out_dir: Path) -> None:
        """
        Point all VLM outputs under *out_dir* (created if missing):
          - <out_dir>/images/bim_render/   (BIM renders)
          - <out_dir>/images/depth/        (ControlNet depth hints)
          - <out_dir>/images/site_photo/   (synthesised site photos)
          - <out_dir>/vlm_training_data.jsonl
        """
        self.output_root = out_dir
        self.bim_render_dir = out_dir / "images" / "bim_render"
        self.site_photo_dir = out_dir / "images" / "site_photo"
        self.depth_map_dir = out_dir / "images" / "depth"
        for d in (self.bim_render_dir, self.site_photo_dir, self.depth_map_dir):
            d.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = out_dir / "vlm_training_data.jsonl"

    def process_renders(
        self,
        render_paths: List[Path],
        elements: List[IFCElementInfo],
        model_id: str,
        project_type: str = "건물",
        trade_type: str = "철근콘크리트",
        depth_paths: Optional[List[Optional[Path]]] = None,
    ) -> int:
        """
        For each BIM render, synthesise a site photo and write a VLMSample.

        *depth_paths* is index-aligned with *render_paths*; where an entry is
        missing the colour render is used as the ControlNet hint instead.

        Returns the number of successfully written samples.
        """
        successful = 0
        elem_ids = [e.global_id for e in elements[:10]]  # cap to 10
        depths = depth_paths or []
        photo_views = self.config.vlm_photo_views
        skipped: Dict[str, int] = {}

        # Sidecar catalog (audit/verification asset) + grounding text for the VLM.
        # Both are derived from element metadata only; image synthesis is untouched.
        self._write_element_catalog(elements, model_id)
        bim_context = self._elements_to_text(elements)

        for index, render_path in enumerate(render_paths):
            view_name = render_path.stem.split("_")[-1]
            depth_path = depths[index] if index < len(depths) else None

            # An orthographic plan or elevation has no photographic equivalent:
            # asked for a "photo" of a top view the sampler stands the slab up
            # and paints it as a facade, which reads as a 90-degree rotation.
            if photo_views and view_name not in photo_views:
                skipped[view_name] = skipped.get(view_name, 0) + 1
                continue

            # UNCHANGED: ComfyUI depth-conditioned site-photo synthesis.
            site_photo_path = self._synthesise_site_photo(
                render_path,
                depth_path=depth_path,
                project_type=project_type,
                trade_type=trade_type,
                view_type=view_name,
            )

            if site_photo_path is None:
                logger.warning("Skipping render %s — site photo synthesis failed", render_path.name)
                continue

            view_type = f"3d_{view_name}" if view_name != "top" else "top_view"
            # One sample per configured task (site-only, bim+site, …). The image
            # pair is shared; only which images/instruction/output differ.
            for sample in self._build_task_samples(
                render_path=render_path,
                site_photo_path=site_photo_path,
                elem_ids=elem_ids,
                bim_context=bim_context,
                project_type=project_type,
                trade_type=trade_type,
                view_type=view_type,
            ):
                self._append_sample(sample)
                successful += 1

        if skipped:
            logger.info(
                "Skipped %d render(s) not in vlm_photo_views=%s: %s",
                sum(skipped.values()), photo_views,
                ", ".join(f"{v}×{n}" for v, n in sorted(skipped.items())),
            )
        return successful

    def _synthesise_site_photo(
        self,
        render_path: Path,
        depth_path: Optional[Path] = None,
        project_type: str = "건물",
        trade_type: str = "철근콘크리트",
        view_type: str = "perspective",
    ) -> Optional[Path]:
        """
        Try ComfyUI synthesis; fall back to a direct copy of the BIM render.
        """
        if self._is_comfyui_available():
            result = self._run_comfyui(
                render_path,
                depth_path=depth_path,
                project_type=project_type,
                trade_type=trade_type,
                view_type=view_type,
            )
            if result:
                return result
            logger.warning("ComfyUI synthesis failed — using BIM render as fallback.")

        # Fallback: copy the BIM render as the site photo
        return self._copy_as_site_photo(render_path)

    def _control_image(self, render_path: Path, depth_path: Optional[Path]) -> Path:
        """Pick the image that will condition ControlNet."""
        if self.config.vlm_control_hint == "depth":
            if depth_path and depth_path.exists():
                return depth_path
            logger.warning(
                "vlm_control_hint='depth' but no depth map for %s — falling back to "
                "the colour render, which conditions far more weakly.",
                render_path.name,
            )
        return render_path

    def _prompts(
        self, project_type: str, trade_type: str, view_type: str
    ) -> Tuple[str, str]:
        """Resolve the configured prompt templates for one image."""
        fields = {
            "project_type": project_type,
            "trade_type": trade_type,
            "view_type": view_type,
        }
        positive = _fmt_prompt(self.config.vlm_positive_prompt, **fields)
        extra = (self.config.vlm_trade_prompts or {}).get(trade_type)
        if extra:
            positive = f"{positive}, {_fmt_prompt(extra, **fields)}"
        negative = _fmt_prompt(self.config.vlm_negative_prompt, **fields)
        return positive, negative

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

        # Prefer the configured name; then any model matching the hint type we
        # actually produce (a depth hint through a canny ControlNet conditions
        # on the wrong signal); then anything installed.
        preferred = self.config.controlnet_model.lower()
        hint = self.config.vlm_control_hint.lower()
        hint_models = [m for m in available if hint in m.lower()]
        if preferred in [m.lower() for m in available]:
            selected_cn = next(m for m in available if m.lower() == preferred)
        elif hint_models:
            selected_cn = hint_models[0]
        else:
            selected_cn = available[0]
            logger.warning(
                "No ControlNet matching hint type '%s' is installed — '%s' will be "
                "conditioned on a signal it was not trained for, so the BIM form "
                "may not be held.",
                hint, selected_cn,
            )

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

        return True

    def _seed_for(self, stem: str) -> int:
        """Configured seed, or a stable per-image one so views stay distinct."""
        if self.config.vlm_seed >= 0:
            return self.config.vlm_seed
        # Deterministic in the image name: reruns reproduce, siblings differ.
        return zlib.crc32(stem.encode("utf-8")) & 0x7FFFFFFF

    def _run_comfyui(
        self,
        render_path: Path,
        depth_path: Optional[Path] = None,
        project_type: str = "건물",
        trade_type: str = "철근콘크리트",
        view_type: str = "perspective",
    ) -> Optional[Path]:
        """Full ComfyUI pipeline: upload → queue → poll → download."""
        try:
            if not self._resolve_models():
                logger.error(
                    "Aborting ComfyUI run — required models are not installed. "
                    "See above for installation instructions."
                )
                return None

            control_path = self._control_image(render_path, depth_path)
            control_name = self._upload_image(control_path)
            if not control_name:
                return None

            init_name = None
            if self.config.vlm_init_from_render:
                init_name = (
                    control_name
                    if control_path == render_path
                    else self._upload_image(render_path)
                )
                if not init_name:
                    return None

            positive, negative = self._prompts(project_type, trade_type, view_type)
            workflow = _build_comfyui_workflow(
                control_image_name=control_name,
                sd_model=self._resolved_checkpoint,
                controlnet_model=self._resolved_controlnet,
                positive=positive,
                negative=negative,
                width=self.config.vlm_image_width,
                height=self.config.vlm_image_height,
                steps=self.config.i2i_steps,
                cfg=self.config.i2i_cfg,
                sampler=self.config.vlm_sampler,
                scheduler=self.config.vlm_scheduler,
                denoise=self.config.i2i_denoise,
                controlnet_strength=self.config.controlnet_strength,
                seed=self._seed_for(render_path.stem),
                control_start=self.config.controlnet_start_percent,
                control_end=self.config.controlnet_end_percent,
                init_image_name=init_name,
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

        # ComfyUI's SaveImage emits PNG; naming it .jpg made every file lie
        # about its format.
        out_path = self.site_photo_dir / f"{stem}_site.png"
        out_path.write_bytes(resp.content)
        logger.info("Site photo saved: %s", out_path.name)
        return out_path

    def _copy_as_site_photo(self, render_path: Path) -> Path:
        """Fallback — copy the BIM render as site photo placeholder."""
        import shutil  # noqa: PLC0415
        out_path = self.site_photo_dir / f"{render_path.stem}_site.png"
        shutil.copy2(render_path, out_path)
        logger.info("Fallback site photo saved (copy): %s", out_path.name)
        return out_path

    # ── Task-driven sample construction ─────────────────────────────────────

    # Which image goes with each task token, as (relative path, absolute path).
    def _image_for(self, token: str, render_path: Path, site_photo_path: Path):
        if token == "bim":
            return f"images/bim_render/{render_path.name}", render_path
        if token == "site":
            return f"images/site_photo/{site_photo_path.name}", site_photo_path
        raise ValueError(f"Unknown vlm_tasks image token: {token!r} (use 'bim'/'site')")

    def _build_task_samples(
        self,
        render_path: Path,
        site_photo_path: Path,
        elem_ids: List[str],
        bim_context: str,
        project_type: str,
        trade_type: str,
        view_type: str,
    ) -> List[VLMSample]:
        """One VLMSample per configured task; schema is unchanged across tasks."""
        samples: List[VLMSample] = []
        for task in self.config.vlm_tasks:
            tokens = task.get("images", ["bim", "site"])
            try:
                pairs = [self._image_for(t, render_path, site_photo_path) for t in tokens]
            except ValueError as exc:
                logger.warning("Skipping task %s: %s", task.get("task_type"), exc)
                continue
            rel_paths = [rel for rel, _ in pairs]
            abs_paths = [ap for _, ap in pairs]

            instruction = _fmt_prompt(
                task.get("instruction", ""),
                project_type=project_type, trade_type=trade_type, view_type=view_type,
            )
            output = self._make_output(
                task_type=task.get("task_type", "vlm_task"),
                image_paths=abs_paths,
                instruction=instruction,
                bim_context=bim_context,
                labels=task.get("labels"),
            )

            self._sample_counter += 1
            samples.append(VLMSample(
                id=f"vlm_{self._sample_counter:06d}",
                task_type=task.get("task_type", "vlm_task"),
                images=rel_paths,
                metadata=VLMMetadata(
                    project_type=project_type,
                    bim_element_ids=elem_ids,
                    trade_type=trade_type,
                    view_type=view_type,
                ),
                instruction=instruction,
                output=output,
            ))
        return samples

    # Verdict vocabulary used when a task does not declare its own.
    _DEFAULT_LABELS = ["match", "partial_match", "mismatch", "unknown"]

    def _make_output(
        self, task_type: str, image_paths: List[Path], instruction: str,
        bim_context: str, labels: Optional[List[str]] = None,
    ) -> VLMOutput:
        """
        Produce output via the configured backend, mirroring the sLLM engine's
        backend choices so the two stay consistent:
          "ollama" (alias "vlm") — Ollama vision model (default)
          "gemini"               — Gemini multimodal API (same key as sLLM)
          "template"/other       — no model; honest-empty placeholder
        llama-server is sLLM-text-only and is not used for VLM vision output.
        """
        backend = self.config.vlm_output_backend
        prompt = self._build_vlm_prompt(instruction, bim_context, labels)
        out: Optional[VLMOutput] = None
        if backend in ("ollama", "vlm"):
            out = self._generate_output_via_ollama(image_paths, prompt)
        elif backend == "gemini":
            out = self._generate_output_via_gemini(image_paths, prompt)
        elif backend not in ("template", "none", ""):
            logger.warning("Unknown vlm_output_backend %r — using template output", backend)

        if out is not None:
            return out
        if backend in ("ollama", "vlm", "gemini"):
            logger.warning("VLM output generation failed for %s — using empty fallback", task_type)
        # Template / fallback: leave a clearly-empty, honest placeholder rather
        # than a fabricated answer. label 'unknown' == not asserted.
        return VLMOutput(answer="", label="unknown", evidence=[])

    def _build_vlm_prompt(
        self, instruction: str, bim_context: str, labels: Optional[List[str]]
    ) -> str:
        """Shared prompt for every VLM backend (label vocabulary injected)."""
        label_opts = "|".join(labels or self._DEFAULT_LABELS)
        return (
            f"{instruction}\n\n"
            "참고용 BIM 요소 속성(정답 판단의 근거로 활용):\n"
            f"{bim_context or '(제공된 속성 없음)'}\n\n"
            "아래 JSON 형식으로만 응답하라 (마크다운/추가 텍스트 없이):\n"
            '{"answer": "<한국어 설명>", '
            f'"label": "<{label_opts}>", '
            '"evidence": ["<근거1>", "<근거2>"]}'
        )

    @staticmethod
    def _parse_vlm_json(raw: str) -> Optional[VLMOutput]:
        """Recover {answer,label,evidence} from a model reply; None on failure."""
        data = _extract_json_object(raw)
        if not data or "answer" not in data:
            logger.warning("VLM reply not parseable as expected JSON: %.120s", raw)
            return None
        evidence = data.get("evidence", [])
        if isinstance(evidence, str):
            evidence = [evidence]
        return VLMOutput(
            answer=str(data.get("answer", "")),
            label=str(data.get("label", "unknown")),
            evidence=[str(e) for e in evidence][:8],
        )

    def _generate_output_via_ollama(
        self, image_paths: List[Path], prompt: str
    ) -> Optional[VLMOutput]:
        """Call the Ollama vision model (/api/chat). None on any failure."""
        try:
            images_b64 = [
                base64.b64encode(p.read_bytes()).decode("ascii") for p in image_paths
            ]
        except OSError as exc:
            logger.warning("Could not read image for VLM call: %s", exc)
            return None
        url = f"{self.config.ollama_base_url.rstrip('/')}/api/chat"
        payload = {
            "model": self.config.vlm_ollama_model,
            "stream": False,
            "format": "json",
            "messages": [{"role": "user", "content": prompt, "images": images_b64}],
            "options": {"temperature": self.config.vlm_output_temperature},
        }
        try:
            resp = requests.post(url, json=payload, timeout=self.config.vlm_output_timeout)
            resp.raise_for_status()
            raw = resp.json().get("message", {}).get("content", "")
        except (requests.RequestException, ValueError) as exc:
            logger.warning("Ollama VLM call failed: %s", exc)
            return None
        return self._parse_vlm_json(raw)

    def _generate_output_via_gemini(
        self, image_paths: List[Path], prompt: str
    ) -> Optional[VLMOutput]:
        """
        Call the Gemini multimodal API — same key/model as the sLLM gemini
        backend (`gemini_api_key`, `gemini_model`). None on any failure.
        """
        try:
            from google import genai            # noqa: PLC0415
            from google.genai import types      # noqa: PLC0415
        except ImportError:
            logger.warning("google-genai not installed; cannot use gemini VLM backend")
            return None
        api_key = self.config.gemini_api_key or os.environ.get("GEMINI_API_KEY", "")
        if not api_key:
            logger.warning("Gemini API key not set (gemini_api_key / GEMINI_API_KEY)")
            return None
        try:
            parts = [
                types.Part.from_bytes(data=p.read_bytes(), mime_type="image/png")
                for p in image_paths
            ]
        except OSError as exc:
            logger.warning("Could not read image for Gemini VLM call: %s", exc)
            return None
        try:
            client = genai.Client(api_key=api_key)
            response = client.models.generate_content(
                model=self.config.gemini_model,
                contents=[*parts, prompt],
                config=types.GenerateContentConfig(
                    temperature=self.config.vlm_output_temperature,
                    response_mime_type="application/json",
                ),
            )
            raw = response.text
        except Exception as exc:  # SDK raises various error types
            logger.warning("Gemini VLM call failed: %s", exc)
            return None
        return self._parse_vlm_json(raw)

    def _elements_to_text(self, elements: List[IFCElementInfo]) -> str:
        """Key=value grounding text for the VLM prompt (not stored in samples)."""
        lines: List[str] = []
        for e in elements[: self.config.vlm_context_max_elements]:
            kv = " | ".join(f"{k}={v}" for k, v in list(e.properties.items())[:8])
            name = e.name or "-"
            lines.append(f"- {e.ifc_type} | GlobalId={e.global_id} | Name={name}"
                         + (f" | {kv}" if kv else ""))
        return "\n".join(lines)

    def _write_element_catalog(self, elements: List[IFCElementInfo], model_id: str) -> None:
        """
        Dump per-IFC element properties to <out_dir>/bim_elements.json.

        A passive audit/verification artifact keyed by GlobalId (joins to each
        sample's metadata.bim_element_ids). Not a rule engine; just a dump.
        """
        if not self.config.vlm_write_bim_catalog:
            return
        catalog = {
            "model_id": model_id,
            "element_count": len(elements),
            "elements": {
                e.global_id: {
                    "ifc_type": e.ifc_type,
                    "name": e.name,
                    "properties": e.properties,
                    "render_path": e.render_path,
                }
                for e in elements
            },
        }
        path = self.output_root / "bim_elements.json"
        try:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(catalog, fh, ensure_ascii=False, indent=2)
            logger.info("Wrote BIM element catalog: %s (%d elements)", path.name, len(elements))
        except OSError as exc:
            logger.warning("Could not write BIM element catalog: %s", exc)

    def _append_sample(self, sample: VLMSample) -> None:
        with open(self.jsonl_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(sample.to_jsonl_dict(), ensure_ascii=False) + "\n")
        logger.info("Appended VLM sample %s", sample.id)
