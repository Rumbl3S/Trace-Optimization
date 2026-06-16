"""Within-dataset failure-forecasting AUC for any saved trajectory file (task vs trace).

  python eval/analyze.py results/trajectories_balanced.json
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))

from agents import _build_openai
from forecast import knn_predict, auc


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        HERE, "results", "trajectories_balanced.json")
    recs = json.load(open(path))
    embed = _build_openai()
    labels = [r["label"] for r in recs]
    srcs = [r["src"] for r in recs]
    task_v = [v.tolist() for v in embed([r["task"] for r in recs])]
    trace_v = [v.tolist() for v in embed([r["trace"] for r in recs])]

    def block(title, idx):
        sub = [labels[i] for i in idx]
        if len(set(sub)) < 2:
            print(f"\n  [{title}] n={len(idx)} — one class, skip"); return
        print(f"\n  [{title}]  n={len(idx)}  success={sum(sub)}/{len(sub)}")
        print(f"  {'k':>4}{'AUC task':>10}{'AUC trace':>11}{'lift':>8}")
        for k in (5, 10):
            tv = [task_v[i] for i in idx]; rv = [trace_v[i] for i in idx]
            at = auc(sub, knn_predict(tv, sub, k)); ar = auc(sub, knn_predict(rv, sub, k))
            print(f"  {k:>4}{at:>10.3f}{ar:>11.3f}{ar - at:>+8.3f}")

    print("=" * 48)
    print(f"  WITHIN-DATASET FORECAST AUC  ({os.path.basename(path)}, n={len(recs)})")
    print("=" * 48)
    block("ALL", list(range(len(recs))))
    for s in sorted(set(srcs)):
        block(f"{s} ONLY", [i for i, x in enumerate(srcs) if x == s])


if __name__ == "__main__":
    main()
