"""MVP go/no-go: do TRACE EMBEDDINGS predict agent failure?

1. Run a lightweight agent attempt on FanOutQA + MuSiQue tasks, capturing each
   trajectory (its reasoning/exploration text) and labelling success/failure with the
   dataset scorer.
2. Embed each trajectory (and, as a baseline, the task alone).
3. LEAVE-ONE-OUT k-NN: predict each trajectory's outcome from its nearest neighbours.
4. Report AUC for trace vs task-only vs chance (0.5).

GO if trace-AUC is meaningfully > 0.5 AND > task-only (the trace carries outcome signal
beyond task difficulty). NO-GO if it's ~chance (embeddings don't capture failure
structure -> the whole 'retrieve similar traces -> forecast' idea has no foundation yet).

  python eval/mvp_failure_forecast.py --fanout 40 --musique 40
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys

HERE = os.path.dirname(__file__)
RS = os.path.abspath(os.path.join(HERE, "..", ".."))          # regime_selector/
sys.path[:0] = [RS, os.path.join(RS, "eval"), os.path.abspath(os.path.join(HERE, ".."))]

import llm
import adaptive
import run_fanoutqa as fq
import run_musique as mu
from _common import score as _score
from demo_embed_compare import haiku, _build_openai           # temp=0 agent + OpenAI embedder
import forecast
from forecast import TrajectoryRecord, knn_predict, auc

llm._ensure_api_key()

TRACE_PROMPT = (
    "{ctx}\n\nTask: {task}\n\nWork through this step by step. As you go, explicitly note "
    "WHAT specific information you look for, what you FIND, and what you CANNOT find in "
    "the context. Then give your final answer on the last line as 'ANSWER: ...'.")


def _attempt(task: str, chunks, words: int):
    ctx = adaptive._select_for_single(task, chunks, words)
    out, tok = haiku(TRACE_PROMPT.format(ctx=ctx, task=task))
    return out, tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fanout", type=int, default=40)
    ap.add_argument("--musique", type=int, default=40)
    ap.add_argument("--words", type=int, default=800, help="context slice per attempt")
    ap.add_argument("--ks", default="3,5,10")
    args = ap.parse_args()

    embed = _build_openai()
    items = [(r, "fanout", lambda a, rec: fq.score(a, rec)["loose"]) for r in fq.load(args.fanout)]
    items += [(r, "musique", lambda a, rec: _score(a, rec.gold)["contains"]) for r in mu.load(args.musique)]
    print(f"generating {len(items)} trajectories (Haiku, {args.words}-word slice)\n", flush=True)

    recs = []
    for i, (rec, src, scorer) in enumerate(items):
        try:
            trace, tok = _attempt(rec.task, rec.context_chunks, args.words)
            sc = scorer(trace, rec)
            recs.append(TrajectoryRecord(task=rec.task, src=src, trace=trace, answer=trace,
                                         score=sc, label=1 if sc >= 0.5 else 0))
        except Exception as e:                                # noqa: BLE001
            print(f"  [{i+1}/{len(items)}] failed: {e!r}", flush=True); continue
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(items)}] done", flush=True)

    if len(recs) < 8:
        print("[too few trajectories]"); return
    labels = [r.label for r in recs]
    base = statistics.mean(labels)

    task_vecs = [v.tolist() for v in embed([r.task for r in recs])]
    trace_vecs = [v.tolist() for v in embed([r.trace for r in recs])]

    # persist so we stop regenerating trajectories on every analysis tweak
    import json
    out = os.path.join(HERE, "results", "trajectories.json")
    with open(out, "w") as f:
        json.dump([{"task": r.task, "src": r.src, "score": r.score, "label": r.label,
                    "trace": r.trace} for r in recs], f)
    print(f"\nsaved {len(recs)} trajectories -> {out}", flush=True)

    ks = [int(x) for x in args.ks.split(",")]

    def block(title, idx):
        sub = [labels[i] for i in idx]
        if len(set(sub)) < 2:
            print(f"\n  [{title}]  n={len(idx)} — one class only, AUC undefined"); return 0.0
        tv = [task_vecs[i] for i in idx]
        rv = [trace_vecs[i] for i in idx]
        print(f"\n  [{title}]  n={len(idx)}  success={sum(sub)}/{len(sub)}")
        print(f"  {'k':>4}{'AUC task-only':>16}{'AUC trace':>13}{'lift':>9}")
        b = 0.0
        for k in ks:
            at = auc(sub, knn_predict(tv, sub, k))
            atr = auc(sub, knn_predict(rv, sub, k))
            b = max(b, atr)
            print(f"  {k:>4}{at:>16.3f}{atr:>13.3f}{atr - at:>+9.3f}")
        return b

    print("\n" + "=" * 60)
    print(f"  FAILURE-FORECASTING MVP  (n={len(recs)} trajectories)")
    print("=" * 60)
    print(f"  base rate (fraction success) = {base:.3f}")
    best = block("ALL (confounded by dataset)", list(range(len(recs))))
    # WITHIN-DATASET controls remove the 'just detect the dataset' shortcut
    within_best = 0.0
    for s in ("fanout", "musique"):
        idx = [i for i, r in enumerate(recs) if r.src == s]
        within_best = max(within_best, block(f"{s} ONLY (controlled)", idx))
    print("\n" + "=" * 60)
    print("VERDICT (chance = 0.500):")
    print(f"  within-dataset best AUC = {within_best:.3f}  (this is the honest number)")
    best = within_best
    if best >= 0.60:
        print(f"  GO — trace embeddings predict outcome above chance (best AUC {best:.3f}).")
        print("       The 'retrieve similar traces -> forecast' premise holds; pursue the")
        print("       representation-learning step next.")
    elif best >= 0.55:
        print(f"  WEAK — some signal (best AUC {best:.3f}) but not strong; richer trace")
        print("       representations needed before scaling.")
    else:
        print(f"  NO-GO — trace embeddings ~chance (best AUC {best:.3f}). Surface-text")
        print("       embeddings don't capture failure structure; need a different trace")
        print("       representation before the forecasting idea is viable.")


if __name__ == "__main__":
    main()
