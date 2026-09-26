"""Preference pair generation for DPO.

DPO trains on (prompt, chosen, rejected) triples and needs no reward model
(Rafailov et al., 2023). The usual way to build those triples is to sample
several completions and have a judge rank them, which inherits the judge's
mistakes.

This engine does not judge. It derives pairs from SFT samples whose answer is
already grounded in a source clause, and constructs the rejected side by a
stated corruption of that answer:

    unsupported   the same claim with its evidence removed
    fabricated    a plausible but unstated numeric threshold
    overreach     answering where the honest response is to decline

Each rejection is a failure mode the deployed model must avoid, and the
contrast is verifiable against the source rather than asserted by a model.
"""
from __future__ import annotations

import json
import logging
import random
import re
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger("AEC_Pipeline.dpo")

# A measured threshold, not a reference number: the value must carry a unit
# and must not be preceded by 제 (article/paragraph numbering).
_UNITS = r"(?:mm|cm|m|km|%|퍼센트|kg|t|톤|MPa|N|kN|℃|도|분|시간|일|개월|년|배|회|명|개|인)"
_NUM = re.compile(r"(?<!제)(?<![0-9.])(\d+(?:\.\d+)?)\s*(?=" + _UNITS + r")")

_DROP_EVIDENCE = (
    "Answer without citing any clause, table or figure — state the conclusion only."
)


def _perturb_number(text: str, rng: random.Random) -> Optional[str]:
    """Return *text* with one number changed, or None if it holds none."""
    matches = list(_NUM.finditer(text))
    if not matches:
        return None
    m = rng.choice(matches)
    value = float(m.group(1))
    changed = value * rng.choice([0.5, 0.75, 1.5, 2.0])
    shown = str(int(changed)) if changed == int(changed) else f"{changed:.1f}"
    return text[:m.start(1)] + shown + text[m.end(1):]


def _strip_citation(text: str) -> str:
    """Remove clause references so the claim stands without support."""
    out = re.sub(r"제\s*\d+\s*조(?:의\d+)?(?:\s*제?\d+\s*항)?", "", text)
    out = re.sub(r"\[[^\]]*\]|\([^)]*조[^)]*\)", "", out)
    return re.sub(r"\s{2,}", " ", out).strip() or text


class DPOEngine:
    """Builds preference pairs from finished SFT samples."""

    def __init__(self, config) -> None:
        self.config = config
        self.jsonl_path = Path(config.output_dir) / "dpo_training_data.jsonl"
        self._rng = random.Random(config.dpo_seed)

    def set_output_dir(self, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = out_dir / "dpo_training_data.jsonl"

    def build_pairs(self, sft_samples: List[dict]) -> int:
        """Write preference pairs for *sft_samples*; return the count."""
        kinds = self.config.dpo_rejection_kinds or []
        if not kinds:
            return 0
        written = 0
        for sample in sft_samples:
            for pair in self._pairs_for(sample, kinds):
                self._append(pair)
                written += 1
        if written:
            logger.info("DPO pairs written: %d -> %s", written, self.jsonl_path)
        return written

    def _pairs_for(self, sample: dict, kinds: List[str]) -> List[dict]:
        output = sample.get("output") or {}
        answer = (output.get("answer") or "").strip()
        if not answer:
            return []
        label = output.get("final_label", "answerable")
        context = ((sample.get("input") or {}).get("context") or "").strip()
        instruction = sample.get("instruction", "")
        prompt = f"{instruction}\n\n{context}" if context else instruction

        pairs: List[dict] = []
        for kind in kinds:
            rejected = self._reject(kind, answer, label)
            if rejected and rejected != answer:
                pairs.append({
                    "id": f"dpo_{sample.get('id', 'x')}_{kind}",
                    "task_type": sample.get("task_type", "regulation_qa"),
                    "rejection_kind": kind,
                    "prompt": prompt,
                    "chosen": answer,
                    "rejected": rejected,
                    "source_doc_ids": sample.get("source_doc_ids", []),
                })
        return pairs

    def _reject(self, kind: str, answer: str, label: str) -> Optional[str]:
        if kind == "unsupported":
            stripped = _strip_citation(answer)
            return stripped if stripped != answer else None
        if kind == "fabricated":
            return _perturb_number(answer, self._rng)
        if kind == "overreach":
            # Only meaningful where declining was the correct behaviour.
            if label != "unanswerable":
                return None
            return self.config.dpo_overreach_text
        logger.warning("Unknown dpo rejection kind: %s", kind)
        return None

    def _append(self, pair: dict) -> None:
        self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.jsonl_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(pair, ensure_ascii=False) + "\n")


__all__ = ["DPOEngine", "_DROP_EVIDENCE"]
