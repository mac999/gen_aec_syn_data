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

from .config import PipelineConfig
from .schemas import DocumentChunk, EvidenceBlock, SFTInput, SFTInputMetadata, SFTOutput, SFTSample

logger = logging.getLogger("AEC_Pipeline.sllm_engine")

# ── Prompt template (multi-QA, both backends) ─────────────────────────────
# Uses {{/}} for literal JSON braces; {var} for substitution variables.

_MULTI_QA_PROMPT = """\
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
        self._chain: Optional[Any] = None      # Ollama LangChain chain
        self._chain_lock = threading.Lock()    # guard lazy init
        self._counter_lock = threading.Lock()  # guard sample counter + file
        self._sample_counter = 0

        self.config.sft_output_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.config.sft_output_dir / "sllm_training_data.jsonl"

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
        last_error: Optional[Exception] = None
        for attempt in range(1, self.config.llm_max_retries + 1):
            try:
                if self.config.llm_backend == "llamaserver":
                    raw = self._call_llamaserver(chunk)
                elif self.config.llm_backend == "gemini":
                    raw = self._call_gemini(chunk)
                else:
                    raw = self._call_ollama(chunk)
                samples = self._parse_multi_output(raw, chunk)
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

    def _call_ollama(self, chunk: DocumentChunk) -> str:
        self._ensure_ollama_chain()
        return self._chain.invoke({
            "doc_id": chunk.doc_id,
            "chunk_index": chunk.chunk_index,
            "n": self.config.qa_per_chunk,
            "text": chunk.text[: self.config.chunk_max_size],
        })

    def _ensure_ollama_chain(self) -> None:
        with self._chain_lock:
            if self._chain is not None:
                return
            logger.info(
                "Initialising Ollama: %s @ %s",
                self.config.ollama_model, self.config.ollama_base_url,
            )
            try:
                from langchain_ollama import OllamaLLM as _Ollama  # noqa: PLC0415
                llm = _Ollama(
                    model=self.config.ollama_model,
                    base_url=self.config.ollama_base_url,
                    temperature=self.config.ollama_temperature,
                )
            except ImportError:
                from langchain_community.llms import Ollama as _Ollama  # noqa: PLC0415
                llm = _Ollama(
                    model=self.config.ollama_model,
                    base_url=self.config.ollama_base_url,
                    temperature=self.config.ollama_temperature,
                )
            from langchain_core.prompts import PromptTemplate  # noqa: PLC0415
            prompt = PromptTemplate(
                template=_MULTI_QA_PROMPT,
                input_variables=["doc_id", "chunk_index", "n", "text"],
            )
            self._chain = prompt | llm
            logger.info("Ollama chain ready.")

    # ── Gemini backend ──────────────────────────────────────────────────────

    def _call_gemini(self, chunk: DocumentChunk) -> str:
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

        prompt_text = _MULTI_QA_PROMPT.format(
            doc_id=chunk.doc_id,
            chunk_index=chunk.chunk_index,
            n=self.config.qa_per_chunk,
            text=chunk.text[: self.config.chunk_max_size],
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

    def _call_llamaserver(self, chunk: DocumentChunk) -> str:
        """POST to llama-server /v1/chat/completions with json_object mode."""
        url = f"{self.config.llama_server_url.rstrip('/')}/v1/chat/completions"
        prompt_text = _MULTI_QA_PROMPT.format(
            doc_id=chunk.doc_id,
            chunk_index=chunk.chunk_index,
            n=self.config.qa_per_chunk,
            text=chunk.text[: self.config.chunk_max_size],
        )
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
        self, raw: str, chunk: DocumentChunk
    ) -> List[SFTSample]:
        raw = raw.strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
        raw = re.sub(r"```\s*$", "", raw, flags=re.MULTILINE)
        raw = raw.strip()

        data: Optional[Dict] = None
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group())
                except json.JSONDecodeError:
                    pass

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
            sample = self._build_sample(pair, chunk)
            if sample:
                samples.append(sample)
        return samples

    def _build_sample(
        self, data: Dict, chunk: DocumentChunk
    ) -> Optional[SFTSample]:
        try:
            inp = data.get("input", {})
            inp_meta = inp.get("metadata", {})
            out = data.get("output", {})

            evidence = [
                EvidenceBlock(
                    doc_id=e.get("doc_id", chunk.doc_id),
                    section=e.get("section", ""),
                )
                for e in out.get("evidence", [])
            ]

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
                    final_label=out.get("final_label", "answerable"),
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
