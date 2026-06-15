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
RS = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path[:0] = [RS, os.path.join(RS, "eval"), os.path.abspath(os.path.join(HERE, ".."))]

import numpy as np
import llm
import adaptive
import run_fanoutqa as fq
import run_musique as mu
from _common import score as _score
from demo_embed_compare import haiku, _build_openai

llm._ensure_api_key()

TRACE_PROMPT = (
    "{ctx}\n\nTask: {task}\n\nWork through this step by step. As you go, explicitly note "
    "WHAT specific information you look for, what you FIND, and what you CANNOT find in "
    "the context. Then give your final answer on the last line as 'ANSWER: ...'.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fanout", type=int, default=50)
    ap.add_argument("--musique", type=int, default=40)
    ap.add_argument("--words", type=int, default=3000, help="fan-out retrieved-context budget")
    args = ap.parse_args()

    embed = _build_openai()

    def embed_retrieve(task, chunks, words):
        chunks = [c for c in chunks if c.strip()]
        if not chunks:
            return ""
        cv = np.asarray(embed(chunks), dtype="float32")
        qv = np.asarray(embed([task]), dtype="float32")[0]
        order = np.argsort(-(cv @ qv))
        picked, used = [], 0
        for j in order:
            w = len(chunks[j].split())
            if picked and used + w > words:
                break
            picked.append(chunks[j]); used += w
        return "\n\n".join(picked)

    recs = []
    fitems = [(r, "fanout", lambda a, rec: fq.score(a, rec)["loose"], True) for r in fq.load(args.fanout)]
    mitems = [(r, "musique", lambda a, rec: _score(a, rec.gold)["contains"], False) for r in mu.load(args.musique)]
    items = fitems + mitems
    print(f"generating {len(items)} trajectories (fanout: embed-retrieve {args.words}w)\n", flush=True)

    for i, (rec, src, scorer, use_embed) in enumerate(items):
        try:
            ctx = (embed_retrieve(rec.task, rec.context_chunks, args.words) if use_embed
                   else adaptive._select_for_single(rec.task, rec.context_chunks, 800))
            trace, _ = haiku(TRACE_PROMPT.format(ctx=ctx, task=rec.task))
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
