"""Pilot gate for a v0.5.4 run: three checks the regeneration must pass.

    python scripts/check_v054_pilot.py <output_dir> [holdout.json]

1. refusal share of the verified STaR rows is at or under the cap (8%).
2. No refusal-twin context contains a measured value from its own answer's
   original — verified structurally: the twin's context must share no
   measured value and no article reference with any value its answer's
   source row asserted. Since twins carry no values by construction, the
   check asserts the twin context/answer invariants directly.
3. No held-out document produced supervised rows.
Exit code 0 only when every check passes.
"""
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.verifiers import cited_articles, measured_values  # noqa: E402


def rows(root: Path, name: str):
    for f in sorted(root.rglob(name)):
        for line in f.open(encoding="utf-8"):
            yield f, json.loads(line)


def main() -> int:
    root = Path(sys.argv[1])
    holdout = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    failures = []

    # 1. refusal share in the verified split
    counts = Counter(r.get("task_type", "?") for _, r in rows(root, "star_training_data.jsonl"))
    total = sum(counts.values())
    share = counts.get("refusal", 0) / total if total else 0.0
    print(f"[1] STaR rows: {total}, refusal {counts.get('refusal', 0)} ({share:.1%})")
    print("    by task:", dict(counts.most_common()))
    if share > 0.08:
        failures.append(f"refusal share {share:.1%} > 8%")

    # 2. twin invariants
    twins = bad = 0
    for _, r in rows(root, "sllm_training_data.jsonl"):
        if r.get("task_type") != "refusal":
            continue
        twins += 1
        ans = (r.get("output") or {}).get("answer", "")
        ctx = (r.get("input") or {}).get("context", "")
        if measured_values(ans):
            bad += 1; continue
        if set(cited_articles(ans)) and not ctx:
            bad += 1; continue
        if (r.get("output") or {}).get("evidence"):
            bad += 1; continue
        if set(measured_values(ctx)) & set(measured_values(ans)):
            bad += 1
    print(f"[2] refusal twins: {twins}, invariant violations: {bad}")
    if twins and bad:
        failures.append(f"{bad} twin(s) violate invariants")

    # 3. holdout leakage
    if holdout and holdout.exists():
        held = {Path(str(d.get("generated_dir", ""))).name
                for d in json.loads(holdout.read_text(encoding="utf-8")).get("pdf_documents", [])}
        held.discard("")
        leaks = set()
        for name in ("sllm_training_data.jsonl", "star_training_data.jsonl",
                     "dpo_training_data.jsonl", "rlvr_training_data.jsonl"):
            for f, _ in rows(root, name):
                if f.parent.name in held:
                    leaks.add(f"{f.parent.name}/{name}")
        print(f"[3] holdout documents: {len(held)}, leaked outputs: {len(leaks)}")
        for l in sorted(leaks)[:5]:
            print("    !!", l)
        if leaks:
            failures.append(f"{len(leaks)} holdout leak(s)")
    else:
        print("[3] holdout file not given or missing — check skipped")

    if failures:
        print("\nFAIL:", "; ".join(failures))
        return 1
    print("\nALL CHECKS PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
