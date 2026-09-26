"""Distractor selection for RAFT-style prompts.

A model trained only on the clause that answers the question learns to trust
whatever it is handed. At run time a retriever returns near-misses, and that
model has no training signal for ignoring them. RAFT (Zhang et al., 2024)
fixes this by putting the answering passage among unrelated ones and training
the model to use the right one.

Distractors are drawn from the same document, so they share vocabulary and
formatting with the golden chunk and cannot be spotted by style alone.
"""
from __future__ import annotations

from typing import List, Sequence

from .schemas import DocumentChunk


def _label(index: int) -> str:
    return f"[Passage {index + 1}]"


def build_context(chunk: DocumentChunk, pool: Sequence[DocumentChunk],
                  n_distractors: int, include_golden: bool = True,
                  max_chars: int = 700) -> str:
    """Render a multi-passage context block around *chunk*.

    With ``include_golden=False`` the answering passage is left out, which is
    what the refusal task needs: every passage is a near-miss, so the correct
    response is to decline.

    Distractors are taken at a stride from the document's other chunks rather
    than from neighbours, since adjacent chunks often continue the same clause
    and would not be distractors at all.
    """
    others: List[DocumentChunk] = [c for c in pool if c.chunk_index != chunk.chunk_index]
    picked: List[DocumentChunk] = []
    if others and n_distractors > 0:
        stride = max(1, len(others) // n_distractors)
        start = chunk.chunk_index % max(1, stride)
        for i in range(start, len(others), stride):
            picked.append(others[i])
            if len(picked) >= n_distractors:
                break

    passages = list(picked)
    if include_golden:
        # Place the golden passage by chunk index so its slot varies across
        # samples; a fixed position would be learnable on its own.
        passages.insert(chunk.chunk_index % (len(picked) + 1), chunk)
    if not passages:
        passages = [chunk] if include_golden else []

    return "\n\n".join(
        f"{_label(i)}\n{p.text[:max_chars]}" for i, p in enumerate(passages))
