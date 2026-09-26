"""RLVR task export.

Reinforcement learning with verifiable rewards needs prompts paired with a
reward that a program can compute, not a model's opinion. This engine writes
that pairing: the prompt, the source the answer must agree with, and the
checks to run against a candidate answer.

The RL loop itself belongs to the training framework. What this pipeline can
supply — and what is usually the hard part — is the verifiable half.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List

from .verifiers import checks_for

logger = logging.getLogger("AEC_Pipeline.rlvr")


class RLVREngine:
    def __init__(self, config) -> None:
        self.config = config
        self.jsonl_path = Path(config.output_dir) / "rlvr_training_data.jsonl"

    def set_output_dir(self, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = out_dir / "rlvr_training_data.jsonl"

    def export(self, samples: List[dict], chunk_text: Dict[int, str]) -> int:
        """Write one RLVR task per sample; return the count."""
        written = 0
        with open(self.jsonl_path, "a", encoding="utf-8") as fh:
            for sample in samples:
                task_type = sample.get("task_type", "regulation_qa")
                source = ((sample.get("input") or {}).get("context") or "").strip()
                if len(source) < 40:
                    source = "\n".join(chunk_text.values())
                if not sample.get("instruction"):
                    continue
                fh.write(json.dumps({
                    "id": f"rlvr_{sample.get('id', 'x')}",
                    "task_type": task_type,
                    "prompt": sample["instruction"],
                    # The reward is computed against this text, so it travels
                    # with the task rather than being looked up at train time.
                    "source": source,
                    "reference_answer": (sample.get("output") or {}).get("answer", ""),
                    "verifiers": checks_for(
                        task_type, self.config.star_task_verifiers),
                    "reward": {"type": "rule", "scale": [0.0, 1.0],
                               "aggregation": "mean"},
                    "source_doc_ids": sample.get("source_doc_ids", []),
                }, ensure_ascii=False) + "\n")
                written += 1
        if written:
            logger.info("RLVR tasks written: %d -> %s", written, self.jsonl_path)
        return written
