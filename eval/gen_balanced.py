"""PART 2 — generate a better-balanced trajectory set.

The first run left FanOutQA at 4/40 success (untestable within-dataset). Here we give the
agent EMBEDDING-RETRIEVED context (top chunks by similarity to the task, generous budget)
so more fan-out attempts succeed -> a testable within-fanout control. Saves to
trajectories_balanced.json; analyze with eval/analyze.py.

  python eval/gen_balanced.py --fanout 50 --musique 40 --words 3000
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))

import _util
from bench import run_fanoutqa as fq
from bench import run_musique as mu
from bench._common import score as _score
from agents import haiku, _build_openai
from pipeline import attempt, make_retriever



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fanout", type=int, default=50)
    ap.add_argument("--musique", type=int, default=40)
    ap.add_argument("--words", type=int, default=3000, help="fan-out retrieved-context budget")
    args = ap.parse_args()

    embed = _build_openai()
    recs = []
    fitems = [(r, "fanout", lambda a, rec: fq.score(a, rec)["loose"], True) for r in fq.load(args.fanout)]
    mitems = [(r, "musique", lambda a, rec: _score(a, rec.gold)["contains"], False) for r in mu.load(args.musique)]
    items = fitems + mitems
    print(f"generating {len(items)} trajectories (fanout: embed-retrieve {args.words}w)\n", flush=True)

    for i, (rec, src, scorer, use_embed) in enumerate(items):
        try:
            if use_embed:
                ctx = make_retriever(rec.context_chunks, embed)(rec.task, args.words)
            else:
                ctx = _util.select_for_single(rec.task, rec.context_chunks, 800)
            trace = attempt(rec.task, ctx, haiku)
            sc = scorer(trace, rec)
            recs.append({"task": rec.task, "src": src, "score": sc,
                         "label": 1 if sc >= 0.5 else 0, "trace": trace})
        except Exception as e:                                # noqa: BLE001
            print(f"  [{i+1}/{len(items)}] failed: {e!r}", flush=True); continue
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(items)}] done", flush=True)

    out = os.path.join(HERE, "results", "trajectories_balanced.json")
    json.dump(recs, open(out, "w"))
    for s in ("fanout", "musique"):
        sub = [r for r in recs if r["src"] == s]
        if sub:
            print(f"  {s}: success {sum(r['label'] for r in sub)}/{len(sub)} "
                  f"= {statistics.mean(r['label'] for r in sub):.2f}")
    print(f"saved {len(recs)} -> {out}")


if __name__ == "__main__":
    main()
