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

import zlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .verifiers import checks_for, has_citation

RETRIEVAL_MODES = ("open_book", "closed_book", "raft")

# {n}, {doc_id}, {chunk_index} are filled for every task; {text} only when the
# mode supplies context. The JSON structure is identical across tasks so one
# parser handles all of them; only the field hints vary, because they are the
# strongest signal the model actually follows (a schema line saying "or empty"
# beat the instruction above it every time in practice).
def _schema(context_hint: str, evidence_hint: str, label_hint: str) -> str:
    return (
        'Respond with valid JSON only, no markdown fence:\n'
        '{{"qa_pairs": [{{"instruction": "<question in Korean>", '
        '"input": {{"context": "' + context_hint + '", '
        '"metadata": {{"project_type": "<건축|교량|터널|도로|댐|기타>", "language": "ko"}}}}, '
        '"output": {{"answer": "<answer in Korean>", '
        '"evidence": ' + evidence_hint + ', '
        '"final_label": "' + label_hint + '"}}, '
        '"domain_tags": ["<tag>", "<tag>"], "source_doc_ids": ["{doc_id}"]}}]}}'
    )


_EVIDENCE = '[{{"doc_id": "{doc_id}", "section": "<clause number exactly as cited in the answer>"}}]'
_CONTEXT = "<the sentences from the passage that support the answer, copied verbatim — never empty>"

# Grounding rules shared by every task that sees source text. They mirror the
# STaR verifiers (src/verifiers.py): a number or clause reference the checks
# cannot find in the source gets the sample rejected, so the prompt demands
# exactly what the checks will look for.
_RULES = (
    "Answer rules:\n"
    "- Each answer is 2-4 complete sentences and self-contained: state the rule, "
    "the condition it applies under, and the consequence or required action.\n"
    "- The section or clause numbers this passage carries are: {sections}. "
    "Name one of them inside the answer text itself, written exactly as "
    "listed (e.g. \"1.2.2에 따르면 ...\", \"제12조의2에 따라 ...\"), and put the "
    "same string in evidence.section. A reference that appears only in "
    "evidence.section does not count. Never cite a number that is not in "
    "the list; if the list says none, cite nothing and leave "
    "evidence.section empty.\n"
    "- Copy numeric values and their units exactly as written in the passage. "
    "Never state a number that does not appear in the passage.\n"
    "- Vary the question style across the {n} pairs (definition, criterion, "
    "procedure, consequence).\n\n"
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
            "answered from it alone.\n\n" + _RULES +
            "final_label: \"answerable\" for definition/criterion/procedure questions; "
            "\"compliant\" or \"non_compliant\" only when the question asks whether a "
            "described action or design meets the clause. Never \"unanswerable\".\n\n"
            "Clause (doc_id={doc_id}, chunk={chunk_index}):\n---\n{text}\n---\n\n"
            + _schema(_CONTEXT, _EVIDENCE,
                      "<answerable|compliant|non_compliant>")
        ),
    },
    {
        "name": "procedure",
        "retrieval": "open_book",
        "weight": 1.0,
        "template": _HEAD + (
            "From the clause below, write {n} question-answer pairs about *procedure*: "
            "the order of steps, who performs them, what triggers each step, and what "
            "must be submitted or recorded.\n\n" + _RULES +
            "Procedure-specific rules:\n"
            "- Write each answer as numbered steps (1. 2. 3.), naming the actor of "
            "every step, then one closing sentence citing the clause.\n"
            "- Procedures outlive individual threshold values, so describe the process; "
            "quote a number only when it is essential and written in the passage.\n"
            "- final_label is always \"answerable\".\n\n"
            "Clause (doc_id={doc_id}, chunk={chunk_index}):\n---\n{text}\n---\n\n"
            + _schema(_CONTEXT, _EVIDENCE, "answerable")
        ),
    },
    {
        "name": "terminology",
        "retrieval": "closed_book",
        "weight": 1.0,
        "template": _HEAD + (
            "Using construction-domain knowledge, write {n} question-answer pairs that "
            "explain technical terms from the domain of document '{doc_id}': what the "
            "term means, the field synonyms and abbreviations for it, and how it differs "
            "from a term it is often confused with.\n\n"
            "Answer rules:\n"
            "- Each answer is 2-4 complete sentences.\n"
            "- No clause text is supplied, so do NOT cite clause numbers and do not "
            "invent them: leave evidence.section empty and input.context empty.\n"
            "- final_label is always \"answerable\".\n\n"
            + _schema("", '[{{"doc_id": "{doc_id}", "section": ""}}]', "answerable")
        ),
    },
    {
        "name": "reasoning",
        "retrieval": "open_book",
        "weight": 1.0,
        "template": _HEAD + (
            "From the clause below, write {n} question-answer pairs that require "
            "*multi-step reasoning*: apply the clause to a concrete situation, compare "
            "two conditions, or derive a consequence.\n\n" + _RULES +
            "Reasoning-specific rules:\n"
            "- Each answer shows the intermediate steps before the conclusion: first "
            "the applicable rule with its clause number, then how the situation meets "
            "or fails each condition, then the conclusion.\n"
            "- Every number used in the steps must come from the passage; do not "
            "compute or assume values the passage does not state.\n"
            "- final_label: \"compliant\" or \"non_compliant\" when the question judges "
            "a described case; otherwise \"answerable\".\n\n"
            "Clause (doc_id={doc_id}, chunk={chunk_index}):\n---\n{text}\n---\n\n"
            + _schema(_CONTEXT, _EVIDENCE,
                      "<answerable|compliant|non_compliant>")
        ),
    },
    {
        "name": "structure",
        "retrieval": "open_book",
        "weight": 0.5,
        "template": _HEAD + (
            "From the clause below, write {n} question-answer pairs about *document "
            "structure*: which article, paragraph or item states a given rule, what a "
            "cross-reference points to, and how the provisions are organised.\n\n" + _RULES +
            "Structure-specific rules:\n"
            "- Answers must name the full reference path exactly as written in the "
            "passage (e.g. 제12조 제2항 제1호, or 3.2.1), and only references that "
            "appear in the passage.\n"
            "- final_label is always \"answerable\".\n\n"
            "Clause (doc_id={doc_id}, chunk={chunk_index}):\n---\n{text}\n---\n\n"
            + _schema(_CONTEXT, _EVIDENCE, "answerable")
        ),
    },
    {
        "name": "refusal",
        "retrieval": "raft",
        "weight": 1.0,
        "template": _HEAD + (
            "Below are several passages. Only some relate to the question you will "
            "write. Produce {n} question-answer pairs where the question is plausible "
            "for this domain but the passages do NOT contain the answer.\n\n"
            "Answer rules:\n"
            "- Begin every answer with the exact phrase "
            "\"제시된 자료만으로는 확인할 수 없습니다.\" and then state in one sentence "
            "what information would be needed to answer.\n"
            "- Never invent facts, clause numbers or values. Do not answer from "
            "general knowledge.\n"
            "- evidence is always the empty array [] and final_label is always "
            "\"unanswerable\".\n"
            "- Leave input.context empty; the pipeline attaches the passages.\n\n"
            "Passages (doc_id={doc_id}, chunk={chunk_index}):\n---\n{text}\n---\n\n"
            + _schema("", "[]", "unanswerable")
        ),
    },
    {
        "name": "grounded_rag",
        "retrieval": "raft",
        "weight": 1.5,
        "template": _HEAD + (
            "Below are several passages; only some are relevant. Write {n} "
            "question-answer pairs answerable from the relevant passages only.\n\n"
            + _RULES +
            "RAG-specific rules:\n"
            "- Each answer starts by naming the supporting passage (e.g. "
            "\"[Passage 2]에 따르면\"), then gives the answer with its clause number.\n"
            "- Ignore the unrelated passages — do not mention their content.\n"
            "- Leave input.context empty; the pipeline attaches the passages.\n"
            "- final_label is always \"answerable\".\n\n"
            "Passages (doc_id={doc_id}, chunk={chunk_index}):\n---\n{text}\n---\n\n"
            + _schema("", _EVIDENCE, "answerable")
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


def eligible_tasks(tasks: List[SFTTask], source: str,
                   overrides: Optional[Dict[str, List[str]]] = None) -> List[SFTTask]:
    """The tasks whose checks *source* can satisfy.

    A chunk with no article or section number -- a form, a table, a notice --
    cannot ground a citation, so a task that verifies citations would reject
    every answer written from it. Those chunks go to the other tasks.
    """
    if has_citation(source):
        return tasks
    kept = [t for t in tasks
            if "citations" not in (list(t.verifiers) or checks_for(t.name, overrides))]
    return kept or tasks


def select_task(tasks: List[SFTTask], index: int, doc_id: str = "") -> SFTTask:
    """Pick a task for chunk *index*, spread by weight without an RNG.

    Deterministic so a rerun produces the same mix; a chunk always maps to the
    same task, which keeps --only-new reruns consistent. The offset per
    document matters because most documents are short: without it every one
    of them would start on the same tasks and the last bands would only be
    reached by long documents.
    """
    if not tasks:
        raise ValueError("no SFT tasks configured")
    total = sum(t.weight for t in tasks)
    offset = zlib.crc32(doc_id.encode("utf-8")) % 10_000 if doc_id else 0
    position = ((index * 997 + offset) % 10_000) / 10_000 * total  # 997: coprime stride
    upto = 0.0
    for task in tasks:
        upto += task.weight
        if position < upto:
            return task
    return tasks[-1]
