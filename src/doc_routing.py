"""Route documents between fine-tuning and retrieval.

Not every document in an AEC corpus should be trained on. Frequently amended
regulations, revision notices and superseded editions are the content where
fine-tuning does damage: the model memorises a threshold that is revised six
months later and then states the stale figure confidently, with no way to
correct it short of retraining. Such documents belong in a retrieval index,
where one reindex replaces the outdated text.

The router scores each document on observable volatility signals — no LLM call
— and assigns a route:

    train       stable enough for SFT/DAPT
    retrieve    volatile or reference-only; emit retrieval records instead
    both        train on it and also index it

Signals are drawn from the filename and the opening text: amendment markers,
document class, issue numbers, effective dates, and form/template wording.
Thresholds are config fields, so a corpus with different conventions can be
retuned without touching this file.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

ROUTES = ("train", "retrieve", "both")

# (name, weight, pattern). Positive weight argues for retrieval.
_SIGNALS = [
    ("amendment", 2.0, re.compile(r"개정|제개정이유|조문별|전부개정|일부개정")),
    ("notice", 1.5, re.compile(r"고시|훈령|예규|공고|지침")),
    ("issue_no", 1.0, re.compile(r"제\s*\d{2,4}\s*-?\s*\d*\s*호")),
    # Nearly every regulation states an effective date somewhere in its body,
    # so on its own this says little; it only sharpens the other signals.
    ("effective_date", 0.5, re.compile(r"시행일|시행\s*\d{4}|부칙")),
    ("form", 1.5, re.compile(r"양식|서식|신청서|통보서|대장|증명서")),
]
# Negative weight argues for training: stable technical content.
_STABLE = [
    ("standard", -2.0, re.compile(r"KDS|KCS|설계기준|표준시방서|해설|배치기준|기술기준")),
    ("manual", -1.5, re.compile(r"매뉴얼|지침서|편람|가이드")),
]


@dataclass
class RouteDecision:
    route: str
    score: float
    signals: List[str] = field(default_factory=list)

    def as_metadata(self) -> Dict[str, object]:
        return {"route": self.route, "volatility_score": round(self.score, 2),
                "signals": self.signals}


def _compiled(signals) -> List[tuple]:
    """Build (name, weight, regex) triples from config entries."""
    out = []
    for item in signals or []:
        try:
            out.append((str(item["name"]), float(item["weight"]),
                        re.compile(str(item["pattern"]))))
        except (KeyError, TypeError, re.error) as exc:
            raise ValueError(f"routing_signals entry invalid: {item} ({exc})") from exc
    return out


def classify(path: Path, head_text: str = "", threshold: float = 2.0,
             both_margin: float = 1.0, signals=None) -> RouteDecision:
    """Score *path* (and its opening text) and pick a route.

    ``threshold`` is the score at which a document is considered volatile.
    Scores within ``both_margin`` below it get "both": borderline documents
    are worth training on and indexing, since neither choice is clearly wrong.
    """
    # A filename says what the document *is*; the body merely mentions things.
    # Standards quote their own amendment history, so a body hit counts half —
    # otherwise every design standard scores as a volatile notice.
    name_text = path.name
    body_text = head_text[:2000]
    score = 0.0
    hits: List[str] = []
    table = _compiled(signals) if signals else _SIGNALS + _STABLE
    for name, weight, pattern in table:
        if pattern.search(name_text):
            score += weight
            hits.append(name)
        elif pattern.search(body_text):
            score += weight * 0.5
            hits.append(f"{name}~")
    if score >= threshold:
        route = "retrieve"
    elif score >= threshold - both_margin:
        route = "both"
    else:
        route = "train"
    return RouteDecision(route, score, hits)


def extract_provenance(path: Path, head_text: str = "") -> Dict[str, Optional[str]]:
    """Pull the fields a retrieval index needs but the SFT path never recorded."""
    haystack = f"{path.name}\n{head_text[:4000]}"

    def first(pattern: str) -> Optional[str]:
        m = re.search(pattern, haystack)
        return m.group(0).strip() if m else None

    return {
        "source_name": path.name,
        "issuing_body": first(r"(국토교통부|행정안전부|고용노동부|환경부|소방청|"
                              r"조달청|산업통상자원부|문화재청|해양수산부|농림축산식품부)"),
        "document_no": first(r"제\s*\d{2,4}\s*-?\s*\d*\s*호"),
        "effective_date": first(r"(19|20)\d{2}\s*[.\-년]\s*\d{1,2}\s*[.\-월]\s*\d{1,2}"),
        "revision_label": first(r"(전부개정|일부개정|폐지제정|제정|개정)"),
        "edition_year": first(r"[(\[](19|20)\d{2}[)\]]"),
    }
