"""Rule-based checks over generated answers.

Both STaR and RLVR need the same thing: a verdict on a generated answer that
is computed, not judged. What makes that possible here is that the source
chunk is available at generation time, so a claim can be checked against the
text it came from.

Each verifier returns a score in [0, 1] and a short reason. They are
deliberately shallow — they catch fabrication and missing grounding, not
semantic correctness, which no rule can settle.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List

_UNITS = (r"(?:mm|cm|m|km|%|퍼센트|kg|t|톤|MPa|N|kN|℃|도|분|시간|일|개월|년|배|회|명|개|인)")
_MEASURED = re.compile(r"(?<!제)(?<![0-9.])(\d+(?:\.\d+)?)\s*(?=" + _UNITS + ")")
# Two citation conventions occur in this corpus: statutes number articles
# ("제3조"), while design standards use dotted section numbers ("232.3.1").
_ARTICLE = re.compile(r"제\s*\d+\s*조(?:의\s*\d+)?|\d+(?:\.\d+){1,3}")
_REFUSAL = re.compile(r"확인할 수 없|답할 수 없|제시된.*없|근거가 없|알 수 없")


@dataclass
class Verdict:
    score: float
    reasons: List[str]

    @property
    def passed(self) -> bool:
        return self.score >= 1.0


def _nums(text: str) -> List[str]:
    return [m.group(1) for m in _MEASURED.finditer(text)]


def verify_numbers(answer: str, source: str) -> Verdict:
    """Every measured value in the answer must appear in the source."""
    claimed = _nums(answer)
    if not claimed:
        return Verdict(1.0, ["no numeric claim"])
    available = set(_nums(source))
    missing = [n for n in claimed if n not in available]
    if missing:
        return Verdict(0.0, [f"value not in source: {', '.join(missing[:3])}"])
    return Verdict(1.0, ["numbers grounded"])


def verify_citations(answer: str, source: str) -> Verdict:
    """Article references in the answer must exist in the source."""
    cited = {m.group(0).replace(" ", "") for m in _ARTICLE.finditer(answer)}
    if not cited:
        return Verdict(0.0, ["no article cited"])
    present = {m.group(0).replace(" ", "") for m in _ARTICLE.finditer(source)}
    unknown = [c for c in cited if c not in present]
    if unknown:
        return Verdict(0.0, [f"article not in source: {', '.join(unknown[:3])}"])
    return Verdict(1.0, ["citations grounded"])


def verify_refusal(answer: str, _source: str = "") -> Verdict:
    """The answer must decline rather than assert."""
    return Verdict(1.0, ["declined"]) if _REFUSAL.search(answer) \
        else Verdict(0.0, ["did not decline"])


def verify_length(answer: str, _source: str = "", minimum: int = 20) -> Verdict:
    return Verdict(1.0, ["length ok"]) if len(answer.strip()) >= minimum \
        else Verdict(0.0, [f"under {minimum} chars"])


VERIFIERS = {
    "numbers": verify_numbers,
    "citations": verify_citations,
    "refusal": verify_refusal,
    "length": verify_length,
}

# Which checks apply to which SFT task. A task absent here gets "length" only.
TASK_VERIFIERS: Dict[str, List[str]] = {
    "regulation_qa": ["length", "numbers", "citations"],
    "procedure": ["length", "numbers"],
    "reasoning": ["length", "numbers"],
    "structure": ["length", "citations"],
    "grounded_rag": ["length", "numbers"],
    "refusal": ["length", "refusal"],
    "terminology": ["length"],
}


def checks_for(task_type: str, overrides: Dict[str, List[str]] | None = None
               ) -> List[str]:
    """Which checks apply to *task_type*, config overrides winning."""
    if overrides and task_type in overrides:
        return list(overrides[task_type])
    return TASK_VERIFIERS.get(task_type, ["length"])


def verify(answer: str, source: str, task_type: str,
           names: List[str] | None = None) -> Verdict:
    """Run the checks for *task_type*; score is their mean."""
    chosen = names or TASK_VERIFIERS.get(task_type, ["length"])
    scores: List[float] = []
    reasons: List[str] = []
    for name in chosen:
        fn = VERIFIERS.get(name)
        if fn is None:
            continue
        v = fn(answer, source)
        scores.append(v.score)
        reasons.extend(f"{name}: {r}" for r in v.reasons)
    return Verdict(sum(scores) / len(scores) if scores else 0.0, reasons)
