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
from fnmatch import fnmatch
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


# Every corpus a route may emit. The names match the engines in pipeline.py;
# "rag" is the retrieval index, the rest are trained on.
CORPORA = ("dapt", "sft", "star", "dpo", "rlvr", "rag")

# What the three legacy routes emit, so a config without corpus_routes keeps
# behaving exactly as it did.
_LEGACY_EMIT = {
    "train":    ("dapt", "sft", "star", "dpo", "rlvr"),
    "both":     ("dapt", "sft", "star", "dpo", "rlvr", "rag"),
    "retrieve": ("rag",),
}


@dataclass
class RouteDecision:
    route: str
    score: float
    signals: List[str] = field(default_factory=list)
    # Which corpora this document feeds. None means "derive from route",
    # which is what the scoring path does; a corpus_routes rule sets it
    # explicitly and the route string becomes the rule's name.
    emit: Optional[tuple] = None

    def emits(self, corpus: str) -> bool:
        """Whether *corpus* should be written for this document."""
        allowed = self.emit if self.emit is not None else _LEGACY_EMIT.get(
            self.route, _LEGACY_EMIT["train"])
        return corpus in allowed

    def as_metadata(self) -> Dict[str, object]:
        return {"route": self.route, "volatility_score": round(self.score, 2),
                "signals": self.signals}


def match_rule(path: Path, rules) -> Optional[RouteDecision]:
    """First matching corpus_routes rule for *path*, or None.

    A rule is {name, when, emit}. ``when`` supports ``glob`` (path pattern,
    matched against the path as given and against its name) and ``regex``
    (filename pattern); an empty ``when`` matches everything, which is how a
    default rule is written. Rules are tried in order, so the default goes
    last. Content-based conditions are added by later stages; a rule carrying
    only those is skipped here rather than treated as a match-all.
    """
    for i, rule in enumerate(rules or []):
        if not isinstance(rule, dict):
            raise ValueError(f"corpus_routes[{i}] must be an object")
        when = rule.get("when") or {}
        unknown = set(when) - {"glob", "regex"}
        if unknown:
            # A condition this stage cannot evaluate must not silently pass.
            continue
        ok = True
        if "glob" in when:
            pat = str(when["glob"])
            ok = ok and (fnmatch(str(path), pat) or fnmatch(path.name, pat)
                         or fnmatch(path.as_posix(), pat))
        if "regex" in when:
            try:
                ok = ok and bool(re.search(str(when["regex"]), path.name))
            except re.error as exc:
                raise ValueError(f"corpus_routes[{i}] regex invalid: {exc}")
        if not ok:
            continue
        emit = rule.get("emit")
        if emit is None:
            raise ValueError(f"corpus_routes[{i}] needs an 'emit' list")
        bad = set(emit) - set(CORPORA)
        if bad:
            raise ValueError(
                f"corpus_routes[{i}] emits unknown corpora: {sorted(bad)}; "
                f"known: {CORPORA}")
        name = str(rule.get("name") or f"rule_{i}")
        return RouteDecision(name, 0.0, [f"rule:{name}"], tuple(emit))
    return None


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

    def _year(text: str) -> Optional[str]:
        m = (re.search(r"[(\[]\s*제?\s*((?:19|20)\d{2})\s*-\s*\d+\s*호", text)
             or re.search(r"[(\[]\s*((?:19|20)\d{2})\s*[)\]]", text)
             or re.search(r"\b((?:19|20)\d{2})\s*년\s*판", text))
        return m.group(1) if m else None

    return {
        "source_name": path.name,
        "issuing_body": first(r"(국토교통부|행정안전부|고용노동부|환경부|소방청|"
                              r"조달청|산업통상자원부|문화재청|해양수산부|농림축산식품부)"),
        "document_no": first(r"제\s*\d{2,4}\s*-?\s*\d*\s*호"),
        "effective_date": first(r"(19|20)\d{2}\s*[.\-년]\s*\d{1,2}\s*[.\-월]\s*\d{1,2}"),
        "revision_label": first(r"(전부개정|일부개정|폐지제정|제정|개정)"),
        # Korean notices carry the year inside the issue number —
        # "(제2023-98호)" — which the bare "(2023)" pattern never matched:
        # the field came back empty for every one of the 18,750 v0.5.3
        # records. Both spellings are accepted, and only the year is kept.
        "edition_year": _year(haystack),
    }
