"""STaR-style filtering of generated samples.

Self-Taught Reasoner (Zelikman et al., 2022) keeps only the generations that
reach a correct answer and trains on those, so the model learns from its own
successes. The usual obstacle is knowing which generations were correct.

Here the source chunk is on hand, so correctness is checked by rule
(src/verifiers.py): a stated threshold must appear in the source, a cited
article must exist in it, a refusal task must actually decline. Samples that
pass are written as a filtered training set; the rest are dropped, with the
failure reason recorded so a reviewer can see what was discarded and why.

This is a filter, not a second generation pass — it reuses the samples the
SFT engine already produced.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List

from .verifiers import verify

logger = logging.getLogger("AEC_Pipeline.star")


class STaREngine:
    def __init__(self, config) -> None:
        self.config = config
        self.jsonl_path = Path(config.output_dir) / "star_training_data.jsonl"
        self.rejects_path = Path(config.output_dir) / "star_rejected.jsonl"

    def set_output_dir(self, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = out_dir / "star_training_data.jsonl"
        self.rejects_path = out_dir / "star_rejected.jsonl"

    def filter_samples(self, samples: List[dict], chunk_text: Dict[int, str]) -> int:
        """Keep samples whose answer verifies against its source; return the count."""
        threshold = self.config.star_min_score
        kept = 0
        rejected = 0
        with open(self.jsonl_path, "a", encoding="utf-8") as ok, \
             open(self.rejects_path, "a", encoding="utf-8") as bad:
            for sample in samples:
                answer = ((sample.get("output") or {}).get("answer") or "").strip()
                if not answer:
                    continue
                # Prefer the context the generator recorded; fall back to the
                # chunk it came from when the model returned an empty context.
                source = ((sample.get("input") or {}).get("context") or "").strip()
                if len(source) < 40:
                    source = "\n".join(chunk_text.values())
                verdict = verify(answer, source, sample.get("task_type", ""))
                record = dict(sample)
                record["verification"] = {
                    "score": round(verdict.score, 3),
                    "reasons": verdict.reasons,
                }
                if verdict.score >= threshold:
                    ok.write(json.dumps(record, ensure_ascii=False) + "\n")
                    kept += 1
                else:
                    bad.write(json.dumps(record, ensure_ascii=False) + "\n")
                    rejected += 1
        if kept or rejected:
            logger.info("STaR: kept %d, rejected %d (min_score=%.2f) -> %s",
                        kept, rejected, threshold, self.jsonl_path)
        return kept
