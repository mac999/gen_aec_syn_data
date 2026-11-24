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

from .verifiers import checks_for, verify

logger = logging.getLogger("AEC_Pipeline.star")


class STaREngine:
    def __init__(self, config) -> None:
        self.config = config
        self.jsonl_path = Path(config.output_dir) / "star_training_data.jsonl"
        self.kept_records: List[dict] = []
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
        passed: List[dict] = []
        with open(self.jsonl_path, "a", encoding="utf-8") as ok, \
             open(self.rejects_path, "a", encoding="utf-8") as bad:
            for sample in samples:
                answer = ((sample.get("output") or {}).get("answer") or "").strip()
                if not answer:
                    continue
                # The sentences the model copied plus the chunk they came
                # from: a section number usually sits in the heading line,
                # which the copied sentences leave out. Only a sample with
                # no chunk index falls back to the whole document.
                inp = sample.get("input") or {}
                own = chunk_text.get((inp.get("metadata") or {}).get("chunk_index"))
                parts = [p for p in ((inp.get("context") or "").strip(), own or "") if p]
                source = "\n".join(parts) if parts else "\n".join(chunk_text.values())
                task_type = sample.get("task_type", "")
                verdict = verify(answer, source, task_type,
                                 checks_for(task_type,
                                            self.config.star_task_verifiers))
                record = dict(sample)
                record["verification"] = {
                    "score": round(verdict.score, 3),
                    "reasons": verdict.reasons,
                }
                if verdict.score >= threshold:
                    passed.append(record)
                else:
                    bad.write(json.dumps(record, ensure_ascii=False) + "\n")
                    rejected += 1

            # Per-document share cap. The v0.5.3 run let refusal rows grow to
            # 18.4% of the verified split (0.1% in v0.5.2) because the refusal
            # check passes almost anything that declines; the trained model
            # then refused 60% of answerable items. The cap bounds a task
            # type's share of the verified output; the deterministic id order
            # keeps reruns reproducible. Capped rows are recorded as rejects
            # with their own reason, not silently dropped.
            caps = getattr(self.config, "star_task_share_cap", None) or {}
            if caps and passed:
                total = len(passed)
                seen: Dict[str, int] = {}
                capped_out = []
                for record in sorted(passed, key=lambda r: r.get("id", "")):
                    t = record.get("task_type", "")
                    limit = caps.get(t)
                    if limit is not None and (seen.get(t, 0) + 1) > max(
                            1, int(total * float(limit))):
                        record["verification"]["reasons"] = (
                            record["verification"]["reasons"]
                            + [f"share cap: {t} > {limit}"])
                        bad.write(json.dumps(record, ensure_ascii=False) + "\n")
                        rejected += 1
                        continue
                    seen[t] = seen.get(t, 0) + 1
                    capped_out.append(record)
                passed = capped_out
            for record in passed:
                ok.write(json.dumps(record, ensure_ascii=False) + "\n")
                kept += 1
            # Exposed for the DPO stage: preference pairs built from rows
            # that failed verification would prefer wrong chosen answers.
            self.kept_records = list(passed)
        if kept or rejected:
            logger.info("STaR: kept %d, rejected %d (min_score=%.2f) -> %s",
                        kept, rejected, threshold, self.jsonl_path)
        return kept
