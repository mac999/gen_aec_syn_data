from __future__ import annotations
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

logger = logging.getLogger("AEC_Pipeline.config")


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
    ifc_max_elements: int = 500  # cap for rendering performance

    # ComfyUI / VLM 
    comfyui_url: str = "http://127.0.0.1:8188"
    comfyui_timeout: int = 300          # seconds to wait per image
    controlnet_model: str = "control_v11p_sd15_mlsd.pth" # "control_v11p_sd15_canny.pth"
    sd_base_model: str = "photon_v1.safetensors" # "v1-5-pruned-emaonly.ckpt"
    i2i_denoise: float = 0.70
    i2i_steps: int = 23
    i2i_cfg: float = 7.5
    controlnet_strength: float = 0.80

    # Processing limits 
    max_samples_per_doc: int = 50
    batch_size: int = 5

    # LLM retry logic 
    llm_max_retries: int = 3

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

    def ensure_output_dirs(self) -> None:
        """Create all required output directories."""
        for d in (
            self.sft_output_dir,
            self.vlm_output_dir,
            self.bim_render_dir,
            self.site_photo_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)
        logger.info("Output directory structure ready.")

    # Serialisation 
    @classmethod
    def load_default(cls, path: str | Path | None = None) -> "PipelineConfig":
        """
        Build a config from the project's config.json if present, otherwise
        fall back to the built-in dataclass defaults.

        path: explicit JSON path; when None, looks for ``config.json`` in the
              project root (the directory containing this ``src/`` package).
        """
        if path is None:
            path = Path(__file__).resolve().parent.parent / "config.json"
        path = Path(path)
        if path.exists():
            logger.info("Loading default config from %s", path)
            return cls.from_json(path)
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
        return cls(**data)

    def to_json(self, config_path: str | Path) -> None:
        data = {
            k: str(v) if isinstance(v, Path) else v
            for k, v in self.__dict__.items()
        }
        with open(config_path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False)
        logger.info("Config saved to %s", config_path)
