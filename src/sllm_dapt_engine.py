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
from typing import Dict, List, Optional

from .config import PipelineConfig
from .schemas import DAPTSample, DocumentChunk

logger = logging.getLogger("AEC_Pipeline.sllm_dapt_engine")

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
        self.config.sft_output_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.config.sft_output_dir / "dapt_training_data.jsonl"

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
        meta = doc_meta or {}
        written = 0
        for chunk in chunks:
            sample = self._build_sample(chunk, meta)
            if sample:
                self._append_sample(sample)
                written += 1
        logger.info("Wrote %d DAPT records from %d chunks.", written, len(chunks))
        return written

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
            section_path=meta.get("section_path", ""),
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


