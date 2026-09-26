"""SFT task registry.

The pipeline originally emitted one task type (regulation QA) with the source
clause always present in the prompt. That teaches extraction from supplied
text and nothing else: a model trained on it cannot answer without retrieval,
and never learns the procedural, structural or refusal behaviour a deployed
assistant needs.

A task here is a prompt template plus a *retrieval mode* that decides what
context the prompt carries:

    open_book    the source chunk is supplied          (reading comprehension)
    closed_book  no chunk; the document is named only  (parametric recall)
    raft         the chunk plus distractor chunks      (robust RAG use)

The split follows the usual division of labour between retrieval and weights:
volatile facts (clause numbers, thresholds) stay retrievable, while stable
behaviour (procedure, terminology, structure, refusal) is trained in.
See RAFT (Zhang et al., 2024) for the distractor formulation.

Tasks are declared in config.json under ``sft_tasks``; this module supplies
the defaults and validates whatever the config provides.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

RETRIEVAL_MODES = ("open_book", "closed_book", "raft")

# {n}, {doc_id}, {chunk_index} are filled for every task; {text} only when the
# mode supplies context. Keep the JSON schema identical across tasks so the
# existing parser handles all of them.
_SCHEMA = (
    'Respond with valid JSON only, no markdown fence:\n'
    '{{"qa_pairs": [{{"instruction": "<question in Korean>", '
    '"input": {{"context": "<supporting text or empty>", '
    '"metadata": {{"project_type": "<건축|교량|터널|도로|댐>", "language": "ko"}}}}, '
    '"output": {{"answer": "<answer in Korean>", '
    '"evidence": [{{"doc_id": "{doc_id}", "section": "<clause reference>"}}], '
    '"final_label": "<compliant|non_compliant|answerable|unanswerable>"}}, '
    '"domain_tags": ["<tag>", "<tag>"], "source_doc_ids": ["{doc_id}"]}}]}}'
)

_HEAD = "You generate Korean AEC training data. Write questions and answers in Korean.\n\n"


@dataclass(frozen=True)
class SFTTask:
    name: str
    retrieval: str
    template: str
    weight: float = 1.0
    # Optional per-task overrides. ``model`` lets a task run on a different
    # LLM than the run default — reasoning on a larger model, terminology on
    # a cheaper one. ``verifiers`` overrides which checks STaR and RLVR apply.
    model: str = ""
    verifiers: tuple = ()

    def renders_context(self) -> bool:
        return self.retrieval in ("open_book", "raft")


# Default task set. Each entry trades a different capability; the weights
# spread chunks across them deterministically (see select_task).
DEFAULT_SFT_TASKS: List[Dict[str, Any]] = [
    {
        "name": "regulation_qa",
        "retrieval": "open_book",
        "weight": 2.0,
        "template": _HEAD + (
            "Read the clause below and write {n} question-answer pairs that can be "
            "answered from it alone. Each answer must cite a clause, table or figure.\n\n"
            "Clause (doc_id={doc_id}, chunk={chunk_index}):\n---\n{text}\n---\n\n" + _SCHEMA
        ),
    },
    {
        "name": "procedure",
        "retrieval": "open_book",
        "weight": 1.0,
        "template": _HEAD + (
            "From the clause below, write {n} question-answer pairs about *procedure*: "
            "the order of steps, who performs them, what triggers each step, and what "
            "must be submitted or recorded. Answer in ordered steps, not a single "
            "sentence. Procedures outlive individual threshold values, so state the "
            "process rather than quoting numbers.\n\n"
            "Clause (doc_id={doc_id}, chunk={chunk_index}):\n---\n{text}\n---\n\n" + _SCHEMA
        ),
    },
    {
        "name": "terminology",
        "retrieval": "closed_book",
        "weight": 1.0,
        "template": _HEAD + (
            "Using construction-domain knowledge, write {n} question-answer pairs that "
            "explain the technical terms appearing in document '{doc_id}': what the term "
            "means, the field synonyms and abbreviations for it, and how it differs from "
            "a term it is often confused with. No clause text is supplied; answer from "
            "domain knowledge and set final_label to 'answerable'.\n\n" + _SCHEMA
        ),
    },
    {
        "name": "reasoning",
        "retrieval": "open_book",
        "weight": 1.0,
        "template": _HEAD + (
            "From the clause below, write {n} question-answer pairs that require "
            "*multi-step reasoning*: apply the clause to a concrete situation, compare "
            "two conditions, or derive a consequence. Each answer must show the "
            "intermediate steps before the conclusion.\n\n"
            "Clause (doc_id={doc_id}, chunk={chunk_index}):\n---\n{text}\n---\n\n" + _SCHEMA
        ),
    },
    {
        "name": "structure",
        "retrieval": "open_book",
        "weight": 0.5,
        "template": _HEAD + (
            "From the clause below, write {n} question-answer pairs about *document "
            "structure*: which article, paragraph or item states a given rule, what a "
            "cross-reference points to, and how the provisions are organised. Answers "
            "must name the reference path.\n\n"
            "Clause (doc_id={doc_id}, chunk={chunk_index}):\n---\n{text}\n---\n\n" + _SCHEMA
        ),
    },
    {
        "name": "refusal",
        "retrieval": "raft",
        "weight": 1.0,
        "template": _HEAD + (
            "Below are several passages. Only some relate to the question you will "
            "write. Produce {n} question-answer pairs where the question is plausible "
            "for this domain but the passages do NOT contain the answer. Each answer "
            "must state honestly that the supplied material does not support an answer. "
            "Set evidence to [] and final_label to 'unanswerable'. Never invent facts.\n\n"
            "Passages (doc_id={doc_id}, chunk={chunk_index}):\n---\n{text}\n---\n\n" + _SCHEMA
        ),
    },
    {
        "name": "grounded_rag",
        "retrieval": "raft",
        "weight": 1.5,
        "template": _HEAD + (
            "Below are several passages; only some are relevant. Write {n} "
            "question-answer pairs answerable from the relevant passages only. Each "
            "answer must first say which passage supports it, then give the answer. "
            "Ignore the unrelated passages — do not mention their content.\n\n"
            "Passages (doc_id={doc_id}, chunk={chunk_index}):\n---\n{text}\n---\n\n" + _SCHEMA
        ),
    },
]


def load_tasks(raw: Any) -> List[SFTTask]:
    """Validate and build the task list from config (or the defaults)."""
    entries = raw if isinstance(raw, list) and raw else DEFAULT_SFT_TASKS
    tasks: List[SFTTask] = []
    for i, item in enumerate(entries):
        if not isinstance(item, dict):
            raise ValueError(f"sft_tasks[{i}] must be an object")
        name = str(item.get("name") or f"task_{i}")
        mode = str(item.get("retrieval", "open_book"))
        if mode not in RETRIEVAL_MODES:
            raise ValueError(
                f"sft_tasks[{i}] '{name}': retrieval must be one of {RETRIEVAL_MODES}")
        template = item.get("template")
        if not template:
            raise ValueError(f"sft_tasks[{i}] '{name}': template is required")
        weight = float(item.get("weight", 1.0))
        if weight <= 0:
            raise ValueError(f"sft_tasks[{i}] '{name}': weight must be > 0")
        verifiers = tuple(item.get("verifiers") or ())
        tasks.append(SFTTask(name, mode, str(template), weight,
                             str(item.get("model") or ""), verifiers))
    return tasks


def select_task(tasks: List[SFTTask], index: int) -> SFTTask:
    """Pick a task for chunk *index*, spread by weight without an RNG.

    Deterministic so a rerun produces the same mix; a chunk always maps to the
    same task, which keeps --only-new reruns consistent.
    """
    if not tasks:
        raise ValueError("no SFT tasks configured")
    total = sum(t.weight for t in tasks)
    position = ((index * 997) % 10_000) / 10_000 * total  # 997: coprime stride
    upto = 0.0
    for task in tasks:
        upto += task.weight
        if position < upto:
            return task
    return tasks[-1]
