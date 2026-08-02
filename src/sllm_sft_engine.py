"""
sLLM synthesis engine — Ollama, llama-server, and Gemini backends.

Method 2: qa_per_chunk > 1  → single LLM call produces N QA pairs per chunk.
Method 3: llm_backend="llamaserver" → OpenAI-compatible API with
          response_format=json_object (grammar-enforced) and
          ThreadPoolExecutor parallel inference.
Method 4: llm_backend="gemini" → Google Gemini API with
          response_mime_type="application/json" (JSON-enforced output).
          Requires: pip install google-genai
          Set GEMINI_API_KEY env var or --gemini-api-key.
"""
from __future__ import annotations

import json
import logging
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

import os

import requests

import string

from .config import (
    DEFAULT_SFT_NEGATIVE_PROMPT_TEMPLATE,
    DEFAULT_SFT_PROMPT_TEMPLATE,
    SFT_TEMPLATE_PLACEHOLDERS,
    PipelineConfig,
)
from .schemas import DocumentChunk, EvidenceBlock, SFTInput, SFTInputMetadata, SFTOutput, SFTSample


def _validate_sft_template(name: str, template: str) -> str:
    """
    Return *template* if it carries the required placeholders, else raise.

    Guards against a user editing the prompt in config.json and dropping a
    ``{text}``/``{doc_id}`` field or mis-escaping a literal JSON brace — which
    would otherwise surface only mid-run as an opaque ``KeyError`` from
    ``str.format``. ``string.Formatter().parse`` reads doubled ``{{ }}`` as
    literal text, so real placeholders are detected without false positives.
    """
    try:
        found = {fname for _, fname, _, _ in string.Formatter().parse(template) if fname}
    except ValueError as exc:  # unbalanced single braces
        raise ValueError(
            f"SFT template '{name}' has malformed braces "
            f"(escape literal JSON braces as {{{{ and }}}}): {exc}"
        ) from exc
    missing = set(SFT_TEMPLATE_PLACEHOLDERS) - found
    if missing:
        need = ", ".join("{" + p + "}" for p in SFT_TEMPLATE_PLACEHOLDERS)
        got = ", ".join("{" + m + "}" for m in sorted(missing))
        raise ValueError(
            f"SFT template '{name}' is missing required placeholder(s): {got}. "
            f"Every template must contain: {need}."
        )
    return template

logger = logging.getLogger("AEC_Pipeline.sllm_engine")

# Reasoning models (qwen3, gpt-oss) may wrap their answer in a <think> block.
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
# Trailing commas before a closing brace/bracket — the one malformation worth
# repairing, since it is a pure syntax slip that does not touch content.
_TRAILING_COMMA_RE = re.compile(r",(\s*[}\]])")


def _iter_brace_spans(text: str):
    """
    Yield every top-level ``{...}`` substring, honouring string literals.

    A plain ``re.search(r"\\{.*\\}", ...)`` grabs from the first brace to the
    last, so a stray ``{`` in reasoning text before the JSON swallows the whole
    reply and it fails to parse. Scanning with real brace-depth, and skipping
    braces inside quoted strings, isolates each candidate object instead.
    """
    depth = 0
    start = -1
    in_str = False
    escaped = False
    for i, ch in enumerate(text):
        if in_str:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    yield text[start:i + 1]


def _extract_json_object(raw: str) -> Optional[Dict]:
    """
    Recover the QA JSON object from a raw LLM reply.

    Order matters: a clean reply is parsed verbatim first, so nothing that
    already parses is altered — this only rescues replies the strict path
    would discard, and never changes accepted content. Strategy:

      1. json.loads on the whole (fence/think-stripped) string.
      2. Each balanced ``{...}`` span, preferring one with ``qa_pairs`` /
         ``instruction``; try it as-is, then with trailing commas removed.
    """
    if not raw:
        return None

    cleaned = _THINK_RE.sub("", raw).strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"```\s*$", "", cleaned, flags=re.MULTILINE).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    candidates = list(_iter_brace_spans(cleaned))
    # A QA object is what we want; fall back to any balanced object.
    candidates.sort(
        key=lambda s: ("qa_pairs" in s or '"instruction"' in s, len(s)),
        reverse=True,
    )
    for span in candidates:
        for attempt in (span, _TRAILING_COMMA_RE.sub(r"\1", span)):
            try:
                return json.loads(attempt)
            except json.JSONDecodeError:
                continue
    return None




class SLLM_SFT_Engine:
    """
    Multi-backend sLLM synthesis engine.

    Backends:
      "ollama"      — LangChain + Ollama (original, sequential or parallel).
      "llamaserver" — Direct HTTP to llama-server /v1/chat/completions with
                      response_format=json_object for grammar-enforced JSON.

    Set config.llm_parallel > 1 to run concurrent workers (most effective
    with llama-server started as: llama-server --parallel <N>).
    """

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self._llm: Optional[Any] = None        # lazily-built Ollama LLM
        self._llm_lock = threading.Lock()      # guard lazy init
        self._counter_lock = threading.Lock()  # guard sample counter + file
        self._sample_counter = 0

        # Resolve and validate the prompt templates once, up front — a bad
        # placeholder in a user-edited config.json should fail loudly here, not
        # silently produce empty output thousands of chunks into a run. Empty
        # config values fall back to the built-in defaults.
        self._pos_template = _validate_sft_template(
            "sft_prompt_template",
            config.sft_prompt_template or DEFAULT_SFT_PROMPT_TEMPLATE,
        )
        self._neg_template = _validate_sft_template(
            "sft_negative_prompt_template",
            config.sft_negative_prompt_template or DEFAULT_SFT_NEGATIVE_PROMPT_TEMPLATE,
        )

        # Output path is set per input file via set_output_dir(); default here
        # keeps the engine usable standalone.
        self.jsonl_path = self.config.sft_output_dir / "sllm_training_data.jsonl"

    # ── Prompt rendering ────────────────────────────────────────────────────

    def _is_negative_chunk(self, chunk: DocumentChunk) -> bool:
        """
        Decide deterministically whether *chunk* yields negative samples.

        A Bresenham-style test spreads ``sft_negative_ratio`` evenly across
        chunk indices (e.g. 0.25 → every 4th chunk) without any RNG, so a run
        is reproducible and a config of 0.0 never produces negatives.
        """
        ratio = self.config.sft_negative_ratio
        if ratio <= 0:
            return False
        if ratio >= 1:
            return True
        i = chunk.chunk_index
        return int((i + 1) * ratio) > int(i * ratio)

    def _render_prompt(self, chunk: DocumentChunk, negative: bool) -> str:
        """Fill the positive or negative template for *chunk*."""
        template = self._neg_template if negative else self._pos_template
        try:
            return template.format(
                doc_id=chunk.doc_id,
                chunk_index=chunk.chunk_index,
                n=self.config.qa_per_chunk,
                text=chunk.text[: self.config.chunk_max_size],
            )
        except (KeyError, IndexError, ValueError) as exc:
            kind = "negative" if negative else "positive"
            raise ValueError(
                f"Failed to render SFT {kind} prompt — check the template's "
                f"placeholders and that literal JSON braces are doubled: {exc}"
            ) from exc

    def set_output_dir(self, out_dir: Path) -> None:
        """Point this engine's JSONL output at *out_dir* (created if missing)."""
        out_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = out_dir / "sllm_training_data.jsonl"

    # ── Public API ──────────────────────────────────────────────────────────

    def process_chunks(self, chunks: List[DocumentChunk]) -> int:
        """Process chunks in parallel; return total samples written."""

        if self.config.llm_backend == "none":
            logger.info("LLM backend is 'none'; skipping chunk processing.")
            return 0

        workers = max(1, self.config.llm_parallel)
        logger.info(
            "Processing %d chunks — backend=%s, parallel=%d, qa_per_chunk=%d",
            len(chunks), self.config.llm_backend, workers, self.config.qa_per_chunk,
        )
        successful = 0
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(self._synthesise_with_retry, c) for c in chunks]
            for future in as_completed(futures):
                for sample in (future.result() or []):
                    self._append_sample(sample)
                    successful += 1
                    if successful >= self.config.max_samples_per_doc:
                        logger.info(
                            "Reached max_samples_per_doc=%d",
                            self.config.max_samples_per_doc,
                        )
                        for f in futures:
                            f.cancel()
                        return successful
        return successful

    # ── Retry wrapper ───────────────────────────────────────────────────────

    def _synthesise_with_retry(self, chunk: DocumentChunk) -> List[SFTSample]:
        negative = self._is_negative_chunk(chunk)
        prompt_text = self._render_prompt(chunk, negative)
        last_error: Optional[Exception] = None
        for attempt in range(1, self.config.llm_max_retries + 1):
            try:
                if self.config.llm_backend == "llamaserver":
                    raw = self._call_llamaserver(prompt_text)
                elif self.config.llm_backend == "gemini":
                    raw = self._call_gemini(prompt_text)
                else:
                    raw = self._call_ollama(prompt_text)
                samples = self._parse_multi_output(raw, chunk, negative)
                if samples:
                    return samples
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "Attempt %d/%d chunk %d '%s': %s",
                    attempt, self.config.llm_max_retries,
                    chunk.chunk_index, chunk.doc_id, exc,
                )
                time.sleep(attempt)
        logger.error(
            "All retries exhausted chunk %d '%s': %s",
            chunk.chunk_index, chunk.doc_id, last_error,
        )
        return []

    # ── Ollama backend ──────────────────────────────────────────────────────

    def _call_ollama(self, prompt_text: str) -> str:
        self._ensure_ollama_llm()
        return self._llm.invoke(prompt_text)

    def _ensure_ollama_llm(self) -> None:
        with self._llm_lock:
            if self._llm is not None:
                return
            logger.info(
                "Initialising Ollama: %s @ %s",
                self.config.ollama_model, self.config.ollama_base_url,
            )
            # Generation limits are set here rather than left to the model's
            # defaults. Ollama otherwise opens the model at its full trained
            # context (262 144 for qwen3), reserving tens of GB of KV cache,
            # and caps output at a length this prompt's nested schema exceeds —
            # a truncated reply is unparseable JSON.
            opts = {
                "num_ctx": self.config.ollama_num_ctx,
                "num_predict": self.config.ollama_num_predict,
            }
            # Grammar-enforced JSON, matching what the llamaserver and gemini
            # backends already ask for.
            if self.config.ollama_json_mode:
                opts["format"] = "json"

            try:
                from langchain_ollama import OllamaLLM as _Ollama  # noqa: PLC0415
                llm = _Ollama(
                    model=self.config.ollama_model,
                    base_url=self.config.ollama_base_url,
                    temperature=self.config.ollama_temperature,
                    **opts,
                )
            except (ImportError, TypeError):
                from langchain_community.llms import Ollama as _Ollama  # noqa: PLC0415
                llm = _Ollama(
                    model=self.config.ollama_model,
                    base_url=self.config.ollama_base_url,
                    temperature=self.config.ollama_temperature,
                    **opts,
                )
            # The prompt is rendered per call (positive/negative varies per
            # chunk), so cache only the LLM and invoke it with the finished text.
            self._llm = llm
            logger.info("Ollama LLM ready.")

    # ── Gemini backend ──────────────────────────────────────────────────────

    def _call_gemini(self, prompt_text: str) -> str:
        """Call Google Gemini API with JSON-enforced output."""
        try:
            from google import genai  # noqa: PLC0415
            from google.genai import types  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "google-genai is required: pip install google-genai"
            ) from exc

        api_key = (
            self.config.gemini_api_key
            or os.environ.get("GEMINI_API_KEY", "")
        )
        if not api_key:
            raise ValueError(
                "Gemini API key not set. Use --gemini-api-key or set GEMINI_API_KEY env var."
            )

        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=self.config.gemini_model,
            contents=prompt_text,
            config=types.GenerateContentConfig(
                temperature=self.config.ollama_temperature,
                response_mime_type="application/json",
            ),
        )
        return response.text

    # ── llama-server backend ────────────────────────────────────────────────

    def _call_llamaserver(self, prompt_text: str) -> str:
        """POST to llama-server /v1/chat/completions with json_object mode."""
        url = f"{self.config.llama_server_url.rstrip('/')}/v1/chat/completions"
        payload = {
            "model": "local",
            "messages": [{"role": "user", "content": prompt_text}],
            "temperature": self.config.ollama_temperature,
            "response_format": {"type": "json_object"},  # grammar-enforced JSON
        }
        resp = requests.post(url, json=payload, timeout=120)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    # ── Output parsing ──────────────────────────────────────────────────────

    def _parse_multi_output(
        self, raw: str, chunk: DocumentChunk, negative: bool = False
    ) -> List[SFTSample]:
        data = _extract_json_object(raw)

        if not data:
            logger.warning(
                "Could not parse JSON for chunk %d of '%s'",
                chunk.chunk_index, chunk.doc_id,
            )
            return []

        # Expected shape: {"qa_pairs": [...]}; tolerate a bare single QA object.
        if "qa_pairs" in data:
            pairs = data["qa_pairs"]
        elif "instruction" in data:
            pairs = [data]
        else:
            logger.warning(
                "Unexpected JSON structure for chunk %d of '%s'",
                chunk.chunk_index, chunk.doc_id,
            )
            return []

        samples: List[SFTSample] = []
        for pair in pairs:
            sample = self._build_sample(pair, chunk, negative)
            if sample:
                samples.append(sample)
        return samples

    def _build_sample(
        self, data: Dict, chunk: DocumentChunk, negative: bool = False
    ) -> Optional[SFTSample]:
        try:
            inp = data.get("input", {})
            inp_meta = inp.get("metadata", {})
            out = data.get("output", {})

            # Negative samples are unanswerable by construction: stamp the label
            # and drop any grounding the model may have hallucinated, so the
            # training signal ("no evidence → decline") stays clean regardless
            # of how well the model followed the negative template.
            if negative:
                evidence: List[EvidenceBlock] = []
                final_label = "unanswerable"
            else:
                evidence = [
                    EvidenceBlock(
                        doc_id=e.get("doc_id", chunk.doc_id),
                        section=e.get("section", ""),
                    )
                    for e in out.get("evidence", [])
                ]
                final_label = out.get("final_label", "answerable")

            sample_id = self._next_id()
            return SFTSample(
                id=sample_id,
                task_type=data.get("task_type", "regulation_qa"),
                domain_tags=data.get("domain_tags", []),
                source_doc_ids=data.get("source_doc_ids", [chunk.doc_id]),
                instruction=data.get("instruction", ""),
                input=SFTInput(
                    context=inp.get("context", chunk.text[:500]),
                    metadata=SFTInputMetadata(
                        project_type=inp_meta.get("project_type", "건축"),
                        language=inp_meta.get("language", "ko"),
                    ),
                ),
                output=SFTOutput(
                    answer=out.get("answer", ""),
                    evidence=evidence,
                    final_label=final_label,
                ),
            )
        except Exception as exc:
            logger.warning("Schema validation failed: %s", exc)
            return None

    def _next_id(self) -> str:
        with self._counter_lock:
            self._sample_counter += 1
            return f"sft_{self._sample_counter:06d}"

    def _append_sample(self, sample: SFTSample) -> None:
        with self._counter_lock:
            with open(self.jsonl_path, "a", encoding="utf-8") as fh:
                fh.write(
                    json.dumps(sample.to_jsonl_dict(), ensure_ascii=False) + "\n"
                )
        logger.info("Appended %s", sample.id)
