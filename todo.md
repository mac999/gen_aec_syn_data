# TODO — User-Comment Handling Log

This file tracks external feedback / feature requests for `gen_aec_syn_data` and
whether each has been handled. Convention: log every actionable user comment
here with its status (`Open` vs `Done`), the request, the chosen approach, and
the files touched. Newest entries first within each section.

---

## Open

_(none)_

---

## Done

### [x] VLM: task-driven datasets + VLM-in-the-loop output (schema-safe)
- **Source:** reviewer feedback (2026-08), refined over discussion.
- **Request evolution:** originally "add BIM element JSON to VLM input." Refined
  after noting (a) `VLMSample` has no input/context field, (b) property text and
  the answer play different roles per `task_type`, and (c) rule-based labels cost
  maintenance. Landed on: per-task generation + VLM-in-the-loop output, no rules.
- **Hard constraint (met):** `VLMSample` schema structure unchanged; BIM render +
  ComfyUI site-photo synthesis unchanged.
- **Delivered (2026-08-02):**
  - `vlm_tasks` (config): each task emits one sample per render with its own
    `images` (`site` only, or `bim`+`site`) and `instruction`. Defaults:
    `site_description` (1 image), `bim_site_comparison` (2 images).
  - `vlm_output_backend:"vlm"`: sample `output` (answer/label/evidence) generated
    by an Ollama vision model (`vlm_ollama_model`, default `qwen2.5vl:7b`) via
    `/api/chat` on the sample's own images. No ruleset. `"template"` = legacy.
  - Per-IFC `bim_elements.json` sidecar (keyed by `GlobalId`) — a cheap passive
    audit/verification dump, NOT a rule engine. Joins to `metadata.bim_element_ids`.
  - BIM properties passed to the VLM as grounding text only (not stored in sample).
- **Known tradeoff:** site photos are synthetic, so VLM `label`s are weak
  supervision; verification is left to the sidecar (optionally a future
  model-as-judge pass — still no rules).
- **Tests:** 27 checks (schema-unchanged, per-task fan-out, image sets, sidecar,
  mocked VLM parse/fallback) + real end-to-end call on an existing image.
- **Files:** `src/config.py`, `src/vlm_engine.py`, `config.json`, README, version.

### [x] SFT negative samples + config-driven, user-editable prompt templates
- **Source:** reviewer feedback (2026-08).
- **Request:** also generate "unanswerable" negative samples so the model learns
  to decline when evidence is insufficient (anti-hallucination); keep the prompt
  user-customizable via a config file rather than piling on CLI flags.
- **Delivered (2026-08-02):**
  - SFT prompt templates now live in `config.json` / `PipelineConfig`
    (`sft_prompt_template`, `sft_negative_prompt_template`), mirroring the
    existing `vlm_positive_prompt` pattern. Empty value → built-in default.
  - New `sft_negative_ratio` (0.0–1.0) deterministically spreads which chunks
    produce negatives (Bresenham-style, no RNG); `0.0` preserves old behavior.
  - Negative samples are forced to `final_label="unanswerable"` with empty
    `evidence`, so the training signal stays clean regardless of model drift.
  - **Caveat 1 (handled):** templates are validated for the required
    placeholders `{n} {doc_id} {chunk_index} {text}` at engine construction
    (fail-fast, clear error); the per-call render is also guarded.
  - **Caveat 2 (handled):** `PipelineConfig.from_json` now ignores unknown keys
    (with a warning) instead of crashing on a stray/renamed field.
- **Tests:** 38 checks pass — validation, ratio distribution, prompt rendering,
  output parsing (positive vs forced-negative), defensive config load.
- **Files:** `src/config.py`, `src/sllm_sft_engine.py`, `config.json`.
