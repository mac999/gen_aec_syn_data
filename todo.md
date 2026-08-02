# TODO — User-Comment Handling Log

This file tracks external feedback / feature requests for `gen_aec_syn_data` and
whether each has been handled. Convention: log every actionable user comment
here with its status (`Open` vs `Done`), the request, the chosen approach, and
the files touched. Newest entries first within each section.

---

## Open

### [ ] VLM input: add BIM element JSON grounding (schema-safe)
- **Source:** reviewer feedback (2026-08).
- **Request:** compose the VLM input as `{render image, site photo, BIM element JSON}`
  instead of prose-only evidence, to improve training efficiency and make the
  evidence verifiable (currently `VLMOutput.evidence` is free prose).
- **Hard constraint:** the existing VLM training-data schema (`VLMSample`) MUST
  NOT change structure.
- **Recommended approach (no schema change) — sidecar catalog + foreign key:**
  - IFC→JSON per element already exists: `IFCElementInfo.properties` built in
    `src/ifc_processor.py` (`_extract_elements` / `_collect_properties`). Today
    only `global_id`s reach the sample; the rich `properties` are discarded.
  - Emit a sidecar catalog per VLM output dir, e.g.
    `output/<ifc>_vlm/bim_elements.json = { global_id: {ifc_type, name, properties, render_path} }`.
  - Keep `VLMSample` byte-identical: `metadata.bim_element_ids` already links
    each sample to its elements (foreign key). The training adapter joins
    `bim_element_ids` → catalog at load time to build the `{render, site_photo,
    BIM element JSON}` input.
  - Evidence can then cite specific `global_id`s (still `List[str]`, no schema
    change), which makes it checkable against the catalog.
- **Status:** DEFERRED (not started) — parked here on 2026-08-02 at user request.

---

## Done

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
