"""Retrieval-record writer.

The SFT and DAPT writers emit text for training. This one emits chunks shaped
for an index: the passage, where it came from, and the provenance a retrieval
system needs to decide whether a hit is still current — issuing body, document
number, effective date, revision label.

No LLM is involved; the records are the chunks plus metadata.
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, List

from .schemas import DocumentChunk

logger = logging.getLogger("AEC_Pipeline.rag")


class RAGEngine:
    def __init__(self, config) -> None:
        self.config = config
        self.jsonl_path = Path(config.output_dir) / "rag_corpus.jsonl"

    def set_output_dir(self, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = out_dir / "rag_corpus.jsonl"

    def process_chunks(self, chunks: List[DocumentChunk],
                       provenance: Dict[str, object],
                       routing: Dict[str, object]) -> int:
        """Write one retrieval record per chunk; return the count."""
        written = 0
        with open(self.jsonl_path, "a", encoding="utf-8") as fh:
            for chunk in chunks:
                text = chunk.text.strip()
                if len(text) < self.config.rag_min_chars:
                    continue
                digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
                fh.write(json.dumps({
                    "id": f"rag_{chunk.doc_id}_{chunk.chunk_index:05d}",
                    "text": text,
                    "content_hash": digest,
                    "doc_id": chunk.doc_id,
                    "chunk_index": chunk.chunk_index,
                    "page_numbers": chunk.page_numbers,
                    "char_count": chunk.char_count,
                    "provenance": provenance,
                    "routing": routing,
                }, ensure_ascii=False) + "\n")
                written += 1
        if written:
            logger.info("RAG records written: %d -> %s", written, self.jsonl_path)
        return written
