"""PART 1 — reasoning-only ablation.

Does the agent's *process* predict failure, or only its (visibly bad) final answer?
Splits each saved trajectory into REASONING (before 'ANSWER:') and ANSWER (from
'ANSWER:' on), embeds each variant, and reports within-dataset leave-one-out AUC for:
task-only | answer-only | reasoning-only | full-trace.

If reasoning-only stays ~ full-trace, the PROCESS carries the signal (the strong claim).
If only answer-only is high, we're just detecting a bad answer (the weak claim).

  python eval/ablation_reasoning.py
"""
from __future__ import annotations

import json
import os
import re
import sys

HERE = os.path.dirname(__file__)
RS = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path[:0] = [RS, os.path.join(RS, "eval"), os.path.abspath(os.path.join(HERE, ".."))]

from demo_embed_compare import _build_openai
from forecast import knn_predict, auc

_ANS = re.compile(r"answer\s*:", re.I)


def split_trace(trace: str):
    m = list(_ANS.finditer(trace))
    if not m:
        return trace.strip(), "[no answer given]"
    cut = m[-1].start()
    reasoning = trace[:cut].strip() or "[no reasoning]"
    answer = trace[cut:].strip() or "[no answer given]"
    return reasoning, answer


def main():
    path = os.path.join(HERE, "results", "trajectories.json")
    recs = json.load(open(path))
    embed = _build_openai()
    labels = [r["label"] for r in recs]
    srcs = [r["src"] for r in recs]
    reasoning, answer = zip(*(split_trace(r["trace"]) for r in recs))

    variants = {
        "task-only": [r["task"] for r in recs],
        "answer-only": list(answer),
        "reasoning-only": list(reasoning),
        "full-trace": [r["trace"] for r in recs],
    }
    vecs = {name: [v.tolist() for v in embed(texts)] for name, texts in variants.items()}

    def block(title, idx):
        sub = [labels[i] for i in idx]
        if len(set(sub)) < 2:
            print(f"\n  [{title}] n={len(idx)} — one class, skip"); return
        print(f"\n  [{title}]  n={len(idx)}  success={sum(sub)}/{len(sub)}")
        print(f"  {'variant':<16}{'AUC@5':>9}{'AUC@10':>9}")
        for name, V in vecs.items():
            sv = [V[i] for i in idx]
            a5 = auc(sub, knn_predict(sv, sub, 5))
            a10 = auc(sub, knn_predict(sv, sub, 10))
            print(f"  {name:<16}{a5:>9.3f}{a10:>9.3f}")

    print("=" * 52)
    print(f"  REASONING-ONLY ABLATION  (n={len(recs)})")
    print("=" * 52)
    block("ALL (confounded)", list(range(len(recs))))
    for s in ("fanout", "musique"):
        block(f"{s} ONLY (controlled)", [i for i, x in enumerate(srcs) if x == s])
    print("\n" + "=" * 52)
    print("READ: if reasoning-only ~ full-trace and >> task-only, the PROCESS")
    print("predicts failure (strong claim). If only answer-only is high, it's")
    print("mostly 'the answer looks wrong' (weak claim).")


if __name__ == "__main__":
    main()
