"""Content-based document labelling for routing.

Regex signals cannot carry a corpus into another language: the current
rules spell out 고시, 훈령, 별지, and an English or Japanese corpus needs the
whole table rewritten. They also read the surface of a document rather than
what it is -- a file named "지침" whose body is an empty form scores as a
stable manual.

An embedding classifier avoids both. Labels are declared in config with a
handful of example passages each; a document is assigned the label whose
example centroid its embedding is closest to. Swapping corpus or language
means swapping the examples, not editing code.

Two details decide whether this works at all, both measured on the v0.5.3
corpus (form vs clause, 60 held-out chunks):

  hand-written prototypes, no task prefix ....... 53%  (chance)
  corpus-drawn examples with the model's prefix . 97%

So the prefix is mandatory for prefix-trained embedders (nomic and the e5
family), and the examples must come from the corpus being routed rather
than being invented. Cost measured at 27 ms per call: 0.1 minutes for 1,958
documents at eight-way parallelism, against 43 hours for the generation
pass the routing feeds.

The classifier reports a margin alongside the label. A document whose top
two labels are within `uncertain_margin` goes to the LLM stage when one is
configured, and is returned as uncertain otherwise. Only those documents
pay the LLM price: on the measured corpus 9 of 60 fell in that band, so a
cascade costs roughly a sixth of what labelling everything with an LLM
would, for the cases where the cheap signal genuinely cannot decide.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

logger = logging.getLogger("AEC_Pipeline.routing_classifier")


# One label name on a line by itself; anything else is treated as a refusal
# to decide, which keeps a chatty model from being read as a confident one.
DEFAULT_LLM_PROMPT = (
    "Classify the document excerpt into exactly one of these categories:\n"
    "{labels}\n\n"
    "Answer with the category name alone, nothing else. If the excerpt fits "
    "none of them, answer: none\n\n"
    "Excerpt:\n---\n{text}\n---\n\nCategory:")


@dataclass
class LabelVerdict:
    label: str
    margin: float
    scores: Dict[str, float]

    @property
    def uncertain(self) -> bool:
        return self.label == ""


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def _centroid(vectors: List[Sequence[float]]) -> List[float]:
    n = len(vectors)
    return [sum(col) / n for col in zip(*vectors)]


class EmbeddingRouter:
    """Nearest-centroid labelling over an Ollama embedding model."""

    def __init__(self, config) -> None:
        spec = getattr(config, "routing_classifier", None) or {}
        self.enabled = bool(spec.get("labels"))
        self.model = spec.get("model", "nomic-embed-text")
        # Prefix-trained embedders collapse to chance without it; harmless
        # for models that ignore it.
        self.prefix = spec.get("prefix", "classification: ")
        self.margin = float(spec.get("uncertain_margin", 0.0))
        self.max_chars = int(spec.get("max_chars", 1200))
        self.base_url = (spec.get("base_url")
                         or getattr(config, "ollama_base_url",
                                    "http://localhost:11434"))
        self.timeout = int(spec.get("timeout", 60))
        self._labels: Dict[str, List[str]] = {
            str(k): [str(x) for x in v] for k, v in (spec.get("labels") or {}).items()}
        self._centroids: Optional[Dict[str, List[float]]] = None

        # Adjudication for the uncertain band. Off unless a model is named;
        # the prompt is a config string for the same reason the SFT prompts
        # are, and {labels} / {text} are filled in here.
        llm = spec.get("llm") or {}
        self.llm_model = str(llm.get("model") or "")
        self.llm_prompt = str(llm.get("prompt_template") or DEFAULT_LLM_PROMPT)
        self.llm_temperature = float(llm.get("temperature", 0.0))
        self.llm_max_chars = int(llm.get("max_chars", 1500))
        self.llm_calls = 0

    # ── embedding ───────────────────────────────────────────────────────
    def _embed(self, text: str) -> List[float]:
        import requests  # noqa: PLC0415
        body = {"model": self.model, "prompt": self.prefix + text[:self.max_chars]}
        r = requests.post(f"{self.base_url.rstrip('/')}/api/embeddings",
                          json=body, timeout=self.timeout)
        r.raise_for_status()
        vec = r.json().get("embedding")
        if not vec:
            raise ValueError(f"embedding model '{self.model}' returned nothing")
        return vec

    def _ensure_centroids(self) -> Dict[str, List[float]]:
        if self._centroids is None:
            self._centroids = {
                name: _centroid([self._embed(t) for t in texts])
                for name, texts in self._labels.items() if texts}
            logger.info("Routing classifier ready: %d label(s) from %d example(s)",
                        len(self._centroids),
                        sum(len(v) for v in self._labels.values()))
        return self._centroids

    # ── public ──────────────────────────────────────────────────────────
    def label(self, text: str) -> Optional[LabelVerdict]:
        """Nearest label for *text*, or None when the classifier is off.

        Returns an empty label when the top two are closer together than
        ``uncertain_margin`` — the caller sends those on rather than acting
        on a coin flip.
        """
        if not self.enabled or not text.strip():
            return None
        try:
            centroids = self._ensure_centroids()
            vec = self._embed(text)
        except Exception as exc:
            # Routing must not take the run down; the caller falls back to
            # the scored path.
            logger.warning("Routing classifier unavailable (%s); "
                           "falling back to scored routing", exc)
            self.enabled = False
            return None
        scores = {name: _cosine(vec, c) for name, c in centroids.items()}
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])
        if len(ranked) == 1:
            return LabelVerdict(ranked[0][0], 1.0, scores)
        gap = ranked[0][1] - ranked[1][1]
        if gap < self.margin and self.llm_model:
            decided = self._ask_llm(text)
            if decided:
                return LabelVerdict(decided, gap, scores)
        if gap < self.margin:
            return LabelVerdict("", gap, scores)
        return LabelVerdict(ranked[0][0], gap, scores)


    def _ask_llm(self, text: str) -> str:
        """Label from the LLM, or "" when it declines or is unreachable."""
        import requests  # noqa: PLC0415
        prompt = self.llm_prompt.format(
            labels="\n".join(f"- {n}" for n in self._labels),
            text=text[:self.llm_max_chars])
        try:
            r = requests.post(
                f"{self.base_url.rstrip('/')}/api/generate",
                json={"model": self.llm_model, "prompt": prompt, "stream": False,
                      "options": {"temperature": self.llm_temperature}},
                timeout=self.timeout)
            r.raise_for_status()
            answer = (r.json().get("response") or "").strip()
        except Exception as exc:
            logger.warning("Routing LLM unavailable (%s); leaving the "
                           "document uncertain", exc)
            return ""
        self.llm_calls += 1
        # Accept only an exact label name: a model that explains itself has
        # not answered the question it was asked.
        first = answer.splitlines()[0].strip().strip(".\"'") if answer else ""
        if first in self._labels:
            return first
        logger.debug("Routing LLM gave no usable label: %r", answer[:80])
        return ""


__all__ = ["EmbeddingRouter", "LabelVerdict", "DEFAULT_LLM_PROMPT"]
