"""Grouping of PipelineConfig fields for the webview's options panel.

Field types are read from the dataclass itself, so a new config field needs
only a name in GROUPS here (or it lands in the catch-all "other" group).
"""
from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path
from typing import Any, Dict, List

# group key -> config field names, in display order
GROUPS: List[tuple] = [
    ("common", ["input_dir", "output_dir", "dataset_mode",
                "max_samples_per_doc", "batch_size"]),
    ("llm", ["llm_backend", "llm_parallel", "llm_max_retries",
             "ollama_model", "ollama_base_url", "ollama_temperature",
             "ollama_num_ctx", "ollama_num_predict", "ollama_json_mode",
             "llama_server_url", "gemini_model", "gemini_api_key"]),
    ("sft", ["qa_per_chunk", "sft_negative_ratio",
             "sft_prompt_template", "sft_negative_prompt_template"]),
    ("dapt", ["dapt_infer_metadata", "dapt_dedupe"]),
    ("pdf", ["chunk_min_size", "chunk_max_size", "chunk_overlap",
             "space_gap_ratio", "ocr_enabled", "ocr_languages", "ocr_dpi",
             "ocr_use_gpu", "ocr_max_pages"]),
    ("ifc", ["ifc_render_width", "ifc_render_height", "ifc_views",
             "ifc_view_angles", "ifc_max_elements",
             "ifc_min_elements_per_group"]),
    ("vlm", ["vlm_output_backend", "vlm_ollama_model", "vlm_ollama_base_url",
             "vlm_output_temperature", "vlm_output_timeout",
             "vlm_context_max_elements", "vlm_write_bim_catalog",
             "vlm_control_hint", "vlm_photo_views", "vlm_image_width",
             "vlm_image_height", "vlm_control_resolution",
             "vlm_depth_ground_plane", "vlm_ground_roughness",
             "vlm_init_from_render", "vlm_sampler", "vlm_scheduler",
             "vlm_seed", "vlm_positive_prompt", "vlm_negative_prompt",
             "vlm_trade_prompts", "vlm_tasks"]),
    ("comfyui", ["comfyui_url", "comfyui_timeout", "controlnet_model",
                 "sd_base_model", "i2i_denoise", "i2i_steps", "i2i_cfg",
                 "controlnet_strength", "controlnet_start_percent",
                 "controlnet_end_percent"]),
]

CHOICES: Dict[str, List[str]] = {
    "dataset_mode": ["sft", "dapt", "both"],
    "llm_backend": ["ollama", "llamaserver", "gemini", "none"],
    "vlm_output_backend": ["ollama", "gemini", "template"],
    "vlm_control_hint": ["depth", "render"],
    "ifc_views": ["perspective", "top", "front", "side"],
    "vlm_photo_views": ["perspective", "top", "front", "side"],
}

MULTILINE = {
    "sft_prompt_template", "sft_negative_prompt_template",
    "vlm_positive_prompt", "vlm_negative_prompt",
}

SECRET = {"gemini_api_key"}


def _kind(name: str, annotation: Any, value: Any) -> str:
    if name in ("input_dir", "output_dir"):
        return "path"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, (dict, list)):
        if name in CHOICES:
            return "multi"
        return "json"
    return "text"


def describe(config) -> List[dict]:
    """Serialise *config* into UI groups of typed, editable fields."""
    values = {f.name: getattr(config, f.name) for f in fields(config)}
    annotations = {f.name: f.type for f in fields(config)}
    seen = set()
    groups: List[dict] = []

    def make_field(name: str) -> dict:
        value = values[name]
        kind = _kind(name, annotations.get(name), value)
        if kind == "json":
            shown = json.dumps(value, ensure_ascii=False, indent=2)
        elif kind == "path":
            shown = str(value)
        else:
            shown = value
        item = {"name": name, "kind": kind, "value": shown}
        if name in CHOICES:
            item["choices"] = CHOICES[name]
        if name in MULTILINE:
            item["multiline"] = True
        if name in SECRET:
            item["secret"] = True
        return item

    for key, names in GROUPS:
        present = [n for n in names if n in values]
        seen.update(present)
        groups.append({"key": key, "fields": [make_field(n) for n in present]})

    rest = [n for n in values if n not in seen]
    if rest:
        groups.append({"key": "other",
                       "fields": [make_field(n) for n in sorted(rest)]})
    return groups


def apply_updates(config, updates: Dict[str, Any]) -> List[str]:
    """Coerce and write *updates* onto *config*. Returns the changed names."""
    values = {f.name: getattr(config, f.name) for f in fields(config)}
    changed: List[str] = []
    for name, raw in (updates or {}).items():
        if name not in values:
            continue
        current = values[name]
        kind = _kind(name, None, current)
        try:
            if kind == "path":
                new = Path(str(raw))
            elif kind == "bool":
                new = raw if isinstance(raw, bool) else str(raw).lower() in ("1", "true", "yes", "on")
            elif kind == "int":
                new = int(raw)
            elif kind == "float":
                new = float(raw)
            elif kind == "json":
                new = json.loads(raw) if isinstance(raw, str) else raw
            elif kind == "multi":
                new = list(raw) if isinstance(raw, list) else [
                    s.strip() for s in str(raw).split(",") if s.strip()]
            else:
                new = str(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name}: {exc}") from exc
        if new != current:
            setattr(config, name, new)
            changed.append(name)
    config.__post_init__()
    return changed
