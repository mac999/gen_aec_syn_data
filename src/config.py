from __future__ import annotations
import json
import logging
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import List

logger = logging.getLogger("AEC_Pipeline.config")

# ── SFT prompt templates ─────────────────────────────────────────────────────
# Stored here (and surfaced in config.json) so users can tune generation without
# code changes, mirroring how vlm_positive_prompt/vlm_negative_prompt are handled.
# Every template MUST keep the placeholders {n}, {doc_id}, {chunk_index}, {text};
# literal JSON braces must stay doubled ({{ }}). SLLM_SFT_Engine validates this on
# start-up. Set the matching config fields to "" to fall back to these defaults.
SFT_TEMPLATE_PLACEHOLDERS = ("n", "doc_id", "chunk_index", "text")

DEFAULT_SFT_PROMPT_TEMPLATE = """\
당신은 AEC(건축·엔지니어링·건설) 분야 전문 데이터 합성기입니다.
아래 문서 청크를 읽고, 건설 법규 LLM 파인튜닝에 적합한 고품질 질문-답변 쌍을 {n}개 생성하세요.

규칙:
- 각 질문은 반드시 주어진 청크의 내용만으로 답할 수 있어야 합니다.
- 각 답변은 특정 조항, 표, 또는 수치를 반드시 인용해야 합니다.
- 질문과 답변은 반드시 한국어로 작성하세요.
- 서로 다른 관점의 질문을 생성하세요 (예: 정의, 절차, 기준, 처벌 등).
- domain_tags는 관련 AEC 도메인 태그 2~4개로 구성하세요 (예: 구조, 안전, 설비, 기계, 전기, 건축, 토목, 소방 등).
- final_label은 내용에 맞게 "compliant", "non_compliant", "answerable", "unanswerable" 중 하나로 설정하세요.
- 유효한 JSON만 응답하세요 — 마크다운 코드 펜스나 추가 텍스트 없이.

JSON 스키마:
{{
  "qa_pairs": [
    {{
      "instruction": "<구체적인 한국어 질문>",
      "input": {{
        "context": "<관련 문서 chunk 또는 조항 전문>",
        "metadata": {{"project_type": "<건축|교량|터널|도로|댐 등>", "language": "ko"}}
      }},
      "output": {{
        "answer": "<정확하고 근거 있는 한국어 답변>",
        "evidence": [
          {{
            "doc_id": "{doc_id}",
            "section": "<조항 또는 절 참조, 예: '제3조 2항'>"
          }}
        ],
        "final_label": "<compliant|non_compliant|answerable|unanswerable>"
      }},
      "domain_tags": ["<태그1>", "<태그2>"],
      "source_doc_ids": ["{doc_id}"]
    }}
  ]
}}

문서 청크 (doc_id={doc_id}, chunk={chunk_index}):
---
{text}
---

JSON 응답:"""

DEFAULT_SFT_NEGATIVE_PROMPT_TEMPLATE = """\
당신은 AEC(건축·엔지니어링·건설) 분야 전문 데이터 합성기입니다.
아래 문서 청크를 읽고, "근거 부족으로 답할 수 없는" 부정(negative) 학습 샘플 {n}개를 생성하세요.
목적: 모델이 주어진 근거만으로 답할 수 없을 때, 답을 지어내지 않고 정직하게 한계를 밝히도록 학습시키는 것입니다.

규칙:
- 각 질문은 이 청크와 같은 건설 도메인의 그럴듯한 질문이되, 청크 내용만으로는 답할 수 없어야 합니다 (필요한 수치·조항·정의가 청크에 없음).
- 답변(answer)은 반드시 "제공된 근거만으로는 답할 수 없다"는 취지로 정직하게 밝히세요. 사실을 지어내지 마세요.
- evidence 배열은 반드시 빈 배열([])로 두세요.
- final_label은 반드시 "unanswerable"로 설정하세요.
- 질문과 답변은 반드시 한국어로 작성하세요.
- 유효한 JSON만 응답하세요 — 마크다운 코드 펜스나 추가 텍스트 없이.

JSON 스키마:
{{
  "qa_pairs": [
    {{
      "instruction": "<청크로 답할 수 없는 구체적인 한국어 질문>",
      "input": {{
        "context": "<관련 문서 chunk 또는 조항 전문>",
        "metadata": {{"project_type": "<건축|교량|터널|도로|댐 등>", "language": "ko"}}
      }},
      "output": {{
        "answer": "<제공된 근거만으로는 답할 수 없다는 정직한 설명>",
        "evidence": [],
        "final_label": "unanswerable"
      }},
      "domain_tags": ["<태그1>", "<태그2>"],
      "source_doc_ids": ["{doc_id}"]
    }}
  ]
}}

문서 청크 (doc_id={doc_id}, chunk={chunk_index}):
---
{text}
---

JSON 응답:"""


@dataclass
class PipelineConfig:
    input_dir: Path = field(default_factory=lambda: Path("./input"))
    output_dir: Path = field(default_factory=lambda: Path("./output"))

    dataset_mode: str = "sft"             # "sft" | "dapt" | "both"

    # LLM backend 
    llm_backend: str = "ollama"           # "ollama" | "llamaserver" | "gemini"
    llm_parallel: int = 1                 # concurrent worker threads
    qa_per_chunk: int = 3                 # QA pairs generated per chunk call

    # Ollama / sLLM 
    ollama_model: str = "llama3:8b-instruct-q4_K_M"
    ollama_base_url: str = "http://localhost:11434"
    ollama_temperature: float = 0.1
    # Set explicitly so behaviour does not depend on a hand-tuned Modelfile:
    # Ollama defaults to the model's full trained context (262 144 for qwen3),
    # which reserves tens of GB of KV cache for a 700-character chunk.
    ollama_num_ctx: int = 16384
    # The multi-QA prompt's nested schema runs to ~5 000 tokens; a lower cap
    # truncates mid-JSON and the reply cannot be parsed.
    ollama_num_predict: int = 8192
    # Grammar-enforced JSON. Off by default: on a reasoning model such as
    # qwen3 the grammar applies to the answer channel while the model spends
    # its budget in the thinking channel, and the reply comes back empty. The
    # prompt already asks for JSON and non-reasoning models honour it, so turn
    # this on only for a model you have checked.
    ollama_json_mode: bool = False

    # Gemini API 
    gemini_api_key: str = ""              # or set GEMINI_API_KEY env var
    gemini_model: str = "gemini-2.5-flash"

    # llama-server
    llama_server_url: str = "http://localhost:8080"

    # Chunking 
    chunk_min_size: int = 100
    chunk_max_size: int = 300
    chunk_overlap: int = 100

    # IFC rendering 
    ifc_render_width: int = 1024
    ifc_render_height: int = 1024
    ifc_views: List[str] = field(
        default_factory=lambda: ["perspective", "top", "front"]
    )
    # (elev, azim) per view. The default perspective sits low, near eye level —
    # a steep bird's-eye angle does not read as a site photograph.
    ifc_view_angles: dict = field(
        default_factory=lambda: {
            "perspective": [18, -55],
            "top": [90, -90],
            "front": [0, -90],
            "side": [0, 0],
        }
    )
    ifc_max_elements: int = 500  # cap for rendering performance
    # Space groups smaller than this are not rendered: a lone wall or slab on
    # empty ground makes a weak training pair. The whole-model group (index 0)
    # is always kept.
    ifc_min_elements_per_group: int = 5

    # ComfyUI / VLM
    comfyui_url: str = "http://127.0.0.1:8188"
    comfyui_timeout: int = 300          # seconds to wait per image
    controlnet_model: str = "control_v11f1p_sd15_depth.pth"
    # Base SD 1.5 renders architecture flatly; a photoreal finetune is what
    # actually makes the output read as a photograph.
    sd_base_model: str = "Realistic_Vision_V6.0_NV_B1_fp16.safetensors"
    i2i_denoise: float = 1.00
    i2i_steps: int = 30
    i2i_cfg: float = 7.0
    controlnet_strength: float = 0.80
    controlnet_start_percent: float = 0.0
    controlnet_end_percent: float = 0.75

    # ── VLM image synthesis ───────────────────────────────────────────────
    # How the BIM geometry is fed to ControlNet:
    #   "depth"  — z-buffer depth map rasterised from the IFC mesh (recommended;
    #              holds the 3-D form while letting the sampler repaint texture)
    #   "render" — the colour BIM render itself (legacy; tends to leak the
    #              wireframe look straight into the output)
    vlm_control_hint: str = "depth"
    # Only these views get a synthesised photo. An orthographic plan/elevation
    # has no photographic equivalent, so the sampler re-reads it as a facade —
    # which is what made "top" outputs look 90 degrees rotated.
    vlm_photo_views: List[str] = field(default_factory=lambda: ["perspective"])
    # Fill empty depth-map pixels with an infinite ground plane at the model
    # base, giving the hint a horizon and a receding floor. Without it the
    # structure floats in a void and the sampler invents whatever fills it.
    vlm_depth_ground_plane: bool = True
    # Undulation added to that ground, as a fraction of the model's depth
    # range. 0 gives a perfectly flat plane, which renders as a paved slab.
    vlm_ground_roughness: float = 0.10
    # False → start from an empty latent so the render's flat shading cannot
    # bleed through. True → img2img from the BIM render (needs i2i_denoise < 1).
    vlm_init_from_render: bool = False
    vlm_image_width: int = 768
    vlm_image_height: int = 768
    vlm_control_resolution: int = 512   # depth-map raster size fed to ControlNet
    vlm_sampler: str = "dpmpp_2m"
    vlm_scheduler: str = "karras"
    vlm_seed: int = -1                  # -1 → derive a fresh seed per image

    # Prompts. {project_type}, {trade_type} and {view_type} are substituted.
    # Lead with the photograph and the concrete. Rebar is kept to a single
    # subordinate mention on purpose — promoting it makes the sampler tile
    # the whole frame with wire mesh.
    # Ground and concrete colour are both stated explicitly: "earth" alone
    # drifts to desert dunes, which then tint the concrete sand-coloured.
    vlm_positive_prompt: str = (
        "photograph of a {trade_type} building under construction on a "
        "{project_type} site, bare grey structural concrete frame, "
        "cool grey board-formed concrete walls and floor slabs, "
        "plywood formwork panels, a few steel props and scaffold towers, "
        "some starter rebar at slab edges, "
        "standing on dark brown compacted earth, damp churned soil with tyre ruts, "
        "patches of gravel and mud, construction dust and dirt, "
        "soft flat overcast daylight, natural muted colours, "
        "realistic photo, sharp focus, high detail, 35mm lens, "
        "architectural documentary photography"
    )
    # Five families matter: the desert drift, paved ground (the analytic ground
    # plane is a smooth surface, so it reads as concrete unless pushed), the
    # CAD look, the demolition drift, and the wire-mesh wallpaper.
    vlm_negative_prompt: str = (
        "desert, sand dunes, beach, arid landscape, sandy ground, "
        "beige concrete, sand-coloured walls, "
        "concrete pavement, paved plaza, asphalt, smooth flat floor, "
        "polished slab, tiled floor, clean swept ground, lawn, grass, "
        "wire mesh, chain link fence, dense grid, lattice, net, cage, "
        "repeating pattern, wall of rebar, forest of steel bars, "
        "clouds, dramatic sky, sunset, sunbeam, lens flare, "
        "ruins, rubble, demolition, collapsed structure, derelict, "
        "abandoned building, war damage, crumbling cracked concrete, "
        "3d render, cgi, rendering, wireframe, blueprint, cad drawing, diagram, "
        "illustration, drawing, painting, cartoon, anime, sketch, "
        "smooth plastic surface, glossy, chrome, metallic sheen, mirror, "
        "people, workers, animals, vehicles, "
        "text, watermark, logo, signature, "
        "floating objects, surreal, fantasy, distorted geometry, "
        "low quality, blurry, jpeg artifacts, oversaturated, neon colours"
    )
    # Optional per-trade prompt fragment appended to the positive prompt.
    vlm_trade_prompts: dict = field(default_factory=dict)

    # ── sLLM SFT prompt customisation ─────────────────────────────────────
    # User-editable QA-generation templates. Empty string → built-in default.
    # Keep placeholders {n} {doc_id} {chunk_index} {text}; double literal braces.
    sft_prompt_template: str = DEFAULT_SFT_PROMPT_TEMPLATE
    sft_negative_prompt_template: str = DEFAULT_SFT_NEGATIVE_PROMPT_TEMPLATE
    # Fraction (0.0–1.0) of chunks that generate "unanswerable" negative samples
    # instead of positive QA pairs. 0.0 disables negatives (backward-compatible).
    sft_negative_ratio: float = 0.0

    # Processing limits
    max_samples_per_doc: int = 50
    batch_size: int = 5

    # LLM retry logic 
    llm_max_retries: int = 3

    # DAPT corpus 
    # Fill source_type/source_org/source_date/project_type/domain_tags/license
    # with one Ollama call per document; they shipped empty otherwise.
    dapt_infer_metadata: bool = True
    dapt_dedupe: bool = True              # drop repeated chunks by raw_hash

    def __post_init__(self) -> None:
        if isinstance(self.input_dir, str):
            self.input_dir = Path(self.input_dir)
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)

    # Derived output sub-paths
    @property
    def sft_output_dir(self) -> Path:
        return self.output_dir / "sft_dataset"

    @property
    def vlm_output_dir(self) -> Path:
        return self.output_dir / "vlm_dataset"

    @property
    def bim_render_dir(self) -> Path:
        return self.vlm_output_dir / "images" / "bim_render"

    @property
    def site_photo_dir(self) -> Path:
        return self.vlm_output_dir / "images" / "site_photo"

    def file_output_dir(self, stem: str, kind: str) -> Path:
        """
        Per-input-file output directory, e.g. ``output/<stem>_sft/``.

        *stem* is the input file name without extension; *kind* is one of
        ``"sft"``, ``"dapt"``, or ``"vlm"``.
        """
        return self.output_dir / f"{stem}_{kind}"

    def ensure_output_dirs(self) -> None:
        """Create the output root. Per-file sub-directories are made on demand."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Output root ready: %s", self.output_dir)

    # Serialisation 
    @classmethod
    def load_default(cls, path: str | Path | None = None) -> "PipelineConfig":
        """
        Build a config from the project's config.json if present, otherwise
        fall back to the built-in dataclass defaults.

        path: explicit JSON path; when None, searches (in order) the current
              working directory and the project root (the directory containing
              this ``src/`` package). The cwd lookup lets a pip-installed CLI
              pick up a ``config.json`` next to where the user runs it.
        """
        if path is not None:
            path = Path(path)
            if path.exists():
                logger.info("Loading default config from %s", path)
                return cls.from_json(path)
        else:
            candidates = (
                Path.cwd() / "config.json",
                Path(__file__).resolve().parent.parent / "config.json",
            )
            for candidate in candidates:
                if candidate.exists():
                    logger.info("Loading default config from %s", candidate)
                    return cls.from_json(candidate)
            path = candidates[0]
        logger.info("Default config '%s' not found; using built-in defaults.", path)
        return cls()

    @classmethod
    def from_json(cls, config_path: str | Path) -> "PipelineConfig":
        with open(config_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        # Convert string paths back to Path objects
        for key in ("input_dir", "output_dir"):
            if key in data:
                data[key] = Path(data[key])
        # Be defensive: silently accepting unknown keys via cls(**data) would
        # raise a TypeError and abort the whole run over a stray/renamed field.
        # Drop unknowns (with a warning) so an old or hand-edited config.json
        # keeps loading; missing keys just fall back to the dataclass defaults.
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            logger.warning(
                "Ignoring %d unknown config key(s): %s",
                len(unknown), ", ".join(sorted(unknown)),
            )
            data = {k: v for k, v in data.items() if k in known}
        return cls(**data)

    def to_json(self, config_path: str | Path) -> None:
        data = {
            k: str(v) if isinstance(v, Path) else v
            for k, v in self.__dict__.items()
        }
        with open(config_path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False)
        logger.info("Config saved to %s", config_path)
