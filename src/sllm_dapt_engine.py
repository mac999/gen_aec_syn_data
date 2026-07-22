"""
sLLM DAPT (Domain-Adaptive Pre-Training) engine.

Converts DocumentChunk objects into pre-training JSONL records using the
schema defined in schemas.DAPTSample — no LLM inference required.

JSONL fields:
  id, doc_id, source_type, source_name, source_org, source_date,
  language, domain_tags, project_type, text, section_path,
  page_range, license, raw_hash
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

from .config import PipelineConfig
from .schemas import DAPTSample, DocumentChunk

logger = logging.getLogger("AEC_Pipeline.sllm_dapt_engine")

# Korean statute/manual headings, used to build section_path.
_CHAPTER_RE = re.compile(r"(제\s*\d+\s*장[^\n]{0,40}|Chapter\s*\d+[^\n]{0,40})")
_ARTICLE_RE = re.compile(r"(제\s*\d+\s*조(?:의\d+)?)")

_DOC_META_PROMPT = """다음은 한 문서의 도입부입니다. 문서 수준 메타데이터를 추출하세요.

규칙:
- 문서에 근거가 없으면 빈 문자열 ""을 쓰세요. 추측하지 마세요.
- source_type: 법령, 시행령, 지침, 매뉴얼, 시방서, 보고서, 기준 중 하나.
- source_org: 발행 기관명.
- source_date: 발행/개정 연도 또는 날짜 (예: "2024").
- project_type: 건축, 토목, 도로, 교량, 터널, 플랜트, 일반 중 하나.
- domain_tags: 2~5개 (구조, 토목, 건축, 안전, 설비, 전기, 소방, BIM, 품질, 환경 등).
- license: 문서에 명시된 이용 조건. 없으면 "".

JSON만 응답하세요:
{{"source_type":"","source_org":"","source_date":"","project_type":"","domain_tags":[],"license":""}}

문서 도입부:
---
{head}
---
"""


def infer_doc_metadata(
    chunks: List[DocumentChunk], config: PipelineConfig
) -> Dict[str, Any]:
    """
    Ask the local Ollama model for document-level metadata, once per document.

    The DAPT schema carries source_type / source_org / source_date /
    project_type / domain_tags / license, but nothing ever populated them, so
    every record shipped with those fields empty. One cheap call fills them for
    the whole document. Returns {} on any failure — the caller keeps its
    defaults rather than writing a guess.
    """
    head = "\n".join(c.text for c in chunks[:6])[:4000]
    payload = {
        "model": config.ollama_model,
        "prompt": _DOC_META_PROMPT.format(head=head),
        "stream": False,
        # Deliberately no "format": "json" — on a reasoning model the grammar
        # applies to the answer channel while the tokens go to the thinking
        # channel, and the reply comes back empty. The prompt asks for JSON and
        # the object is recovered below.
        "options": {
            "temperature": 0.0,
            "num_ctx": config.ollama_num_ctx,
            "num_predict": 2048,
        },
    }
    try:
        resp = requests.post(
            f"{config.ollama_base_url}/api/generate", json=payload, timeout=600
        )
        resp.raise_for_status()
        raw = resp.json().get("response", "")
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            raise ValueError(f"no JSON object in reply ({len(raw)} chars)")
        data = json.loads(match.group())
    except Exception as exc:
        logger.warning("Document metadata inference failed: %s", exc)
        return {}

    allowed = {
        "source_type", "source_org", "source_date",
        "project_type", "domain_tags", "license",
    }
    meta = {k: v for k, v in data.items() if k in allowed and v}
    if isinstance(meta.get("domain_tags"), str):
        meta["domain_tags"] = [meta["domain_tags"]]
    logger.info("Inferred document metadata: %s", meta)
    return meta

class SLLM_DAPT_Engine:
    """
    Converts DocumentChunk objects to DAPT pre-training JSONL records.

    Document-level metadata (source_type, source_name, source_org,
    source_date, domain_tags, project_type, section_path, license)
    can be supplied via the ``doc_meta`` dict when calling
    ``process_chunks()``.  Any omitted key falls back to an empty
    string / empty list.
    """

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self._sample_counter = 0
        self._chapter = ""
        self._article = ""
        # Output path is set per input file via set_output_dir(); default here
        # keeps the engine usable standalone.
        self.jsonl_path = self.config.sft_output_dir / "dapt_training_data.jsonl"

    def set_output_dir(self, out_dir: Path) -> None:
        """Point this engine's JSONL output at *out_dir* (created if missing)."""
        out_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = out_dir / "dapt_training_data.jsonl"

    # ── Public API ──────────────────────────────────────────────────────────

    def process_chunks(
        self,
        chunks: List[DocumentChunk],
        doc_meta: Optional[Dict] = None,
    ) -> int:
        """
        Convert *chunks* to DAPT records and append to the JSONL file.

        Parameters
        ----------
        chunks:   DocumentChunk list produced by PDFExtractor.
        doc_meta: Optional document-level metadata dict with any of:
                    source_type, source_name, source_org, source_date,
                    language, domain_tags, project_type, section_path,
                    license

        Returns total records written.
        """
        meta = dict(doc_meta or {})

        if self.config.dapt_infer_metadata and chunks:
            for key, value in infer_doc_metadata(chunks, self.config).items():
                meta.setdefault(key, value)

        written = 0
        duplicates = 0
        seen_hashes: set[str] = set()
        self._chapter = ""
        self._article = ""

        for chunk in chunks:
            sample = self._build_sample(chunk, meta)
            if not sample:
                continue
            # raw_hash existed purely as a field; using it here keeps repeated
            # boilerplate (notices, repeated headers) out of the corpus.
            if self.config.dapt_dedupe:
                if sample.raw_hash in seen_hashes:
                    duplicates += 1
                    continue
                seen_hashes.add(sample.raw_hash)
            self._append_sample(sample)
            written += 1

        if duplicates:
            logger.info("Skipped %d duplicate chunk(s) by raw_hash", duplicates)
        logger.info("Wrote %d DAPT records from %d chunks.", written, len(chunks))
        return written

    def _track_section(self, text: str) -> str:
        """Running chapter/article path, carried forward across chunks."""
        chapters = _CHAPTER_RE.findall(text)
        if chapters:
            self._chapter = re.sub(r"\s+", " ", chapters[-1]).strip()
        articles = _ARTICLE_RE.findall(text)
        if articles:
            self._article = re.sub(r"\s+", " ", articles[-1]).strip()
        return " > ".join(p for p in (self._chapter, self._article) if p)

    def _build_sample(
        self, chunk: DocumentChunk, meta: Dict
    ) -> Optional[DAPTSample]:
        text = chunk.text.strip()
        if not text:
            return None

        page_nums = chunk.page_numbers
        if page_nums:
            page_range = (
                str(page_nums[0])
                if len(page_nums) == 1
                else f"{page_nums[0]}-{page_nums[-1]}"
            )
        else:
            page_range = ""

        raw_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        section_path = meta.get("section_path") or self._track_section(text)

        return DAPTSample(
            id=self._next_id(),
            doc_id=chunk.doc_id,
            source_type=meta.get("source_type", ""),
            source_name=meta.get("source_name", chunk.doc_id),
            source_org=meta.get("source_org", ""),
            source_date=meta.get("source_date", ""),
            language=meta.get("language", "ko"),
            domain_tags=meta.get("domain_tags", []),
            project_type=meta.get("project_type", ""),
            text=text,
            section_path=section_path,
            page_range=page_range,
            license=meta.get("license", ""),
            raw_hash=raw_hash,
        )

    def _next_id(self) -> str:
        self._sample_counter += 1
        return f"dapt_{self._sample_counter:06d}"

    def _append_sample(self, sample: DAPTSample) -> None:
        with open(self.jsonl_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(sample.to_jsonl_dict(), ensure_ascii=False) + "\n")
        logger.debug("Appended %s (doc=%s)", sample.id, sample.doc_id)


