"""MuSiQue loader — multi-hop, long-context, dispersed evidence. This is the
escalation-rich benchmark where within-task switching should actually pay off:
each question's paragraph pool mixes a few supporting paragraphs with many
distractors, so single-agent context utilization degrades on the harder instances.

Run standalone to preview records:  python eval/run_musique.py --n 3 --dry-run
"""
from __future__ import annotations

from bench._common import Record, try_load


def _load_hf(n: int) -> list[Record]:
    from datasets import load_dataset
    # 2-hop dev split is small and cheap; answerable subset has clean golds.
    ds = load_dataset("dgslibisey/MuSiQue", split="validation")
    recs = []
    for ex in ds:
        if not ex.get("answerable", True):
            continue
        paras = ex.get("paragraphs", [])
        chunks = [p.get("paragraph_text", "") for p in paras if p.get("paragraph_text")]
        if not chunks:
            continue
        recs.append(Record(task=ex["question"], context_chunks=chunks,
                           gold=ex["answer"],
                           meta={"hops": len(ex.get("question_decomposition", []))}))
        if len(recs) >= n:
            break
    return recs


def load(n: int = 20, dry_run: bool = False) -> list[Record]:
    return try_load(_load_hf, n, dry_run)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    for r in load(args.n, args.dry_run):
        print(f"\nQ: {r.task}\n  chunks={len(r.context_chunks)} gold={r.gold!r} "
              f"hops={r.meta.get('hops')}")
