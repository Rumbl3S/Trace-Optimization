"""PER-COMPONENT forecasting — GENERIC, no benchmark-specific logic.

The unit of prediction becomes a single CHECKABLE COMPONENT of a task, not the whole task.
Everything here works on ANY task; the only task-specific input is a VERIFIER (here a
generic LLM judge against the task's gold string — in deployment, the user's own check).

Per task:
  decompose(task)            -> atomic sub-questions          [generic prompt]
  attempt(subq, retrieved)   -> component trace + answer      [generic]
  verify(subq, answer)       -> 0/1                           [PLUGGABLE; here: gold judge]
=> many (component_trace, label) records -> forecast per component (k-NN), within + across
   task types. No reference to FanOutQA/MuSiQue structure anywhere in the method.

  python eval/component_forecast.py --tasks 18 --max-components 6
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(__file__)
RS = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path[:0] = [RS, os.path.join(RS, "eval"), os.path.abspath(os.path.join(HERE, ".."))]

import numpy as np
import llm
import run_fanoutqa as fq
import run_musique as mu
from demo_embed_compare import haiku, _build_openai
from forecast import knn_predict, knn_predict_cross, auc, spearman

llm._ensure_api_key()

# ── generic operations (no dataset logic) ───────────────────────────────────────
DECOMPOSE = ("Break the task into the minimal list of ATOMIC, independently-checkable "
             "sub-questions whose answers together fully answer it. One sub-question per "
             "line, no numbering, no commentary.\n\nTask: {task}")
ATTEMPT = ("{ctx}\n\nQuestion: {q}\n\nReason step by step, noting what you look for and "
           "what you find, then end with 'ANSWER: ...'.")
JUDGE = ("Gold answer to the overall task:\n{gold}\n\nSub-question: {q}\nProposed answer: "
         "{a}\n\nIs the proposed answer correct and supported by the gold? Reply YES or NO.")


def decompose(task, agent, cap):
    out, _ = agent(DECOMPOSE.format(task=task))
    qs = [ln.strip(" -*\t").strip() for ln in out.splitlines() if len(ln.strip()) > 6]
    return qs[:cap] or [task]


def gold_judge(gold, agent):
    def verify(q, a):
        out, _ = agent(JUDGE.format(gold=gold[:4000], q=q, a=a[:1500]))
        return 1 if "yes" in out.strip().lower()[:5] else 0
    return verify


def make_retriever(chunks, embed):
    chunks = [c for c in chunks if c.strip()]
    cv = np.asarray(embed(chunks), dtype="float32") if chunks else None
    def retrieve(q, words=1200):
        if cv is None:
            return ""
        qv = np.asarray(embed([q]), dtype="float32")[0]
        order = np.argsort(-(cv @ qv))
        picked, used = [], 0
        for j in order:
            w = len(chunks[j].split())
            if picked and used + w > words:
                break
            picked.append(chunks[j]); used += w
        return "\n\n".join(picked)
    return retrieve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", type=int, default=18)
    ap.add_argument("--max-components", type=int, default=6)
    args = ap.parse_args()
    embed = _build_openai()

    sources = [("fanout", fq.load(args.tasks)), ("musique", mu.load(args.tasks))]
    records = []
    for src, recs in sources:
        for ti, rec in enumerate(recs):
            try:
                subqs = decompose(rec.task, haiku, args.max_components)
                retrieve = make_retriever(rec.context_chunks, embed)
                verify = gold_judge(rec.gold, haiku)
                for q in subqs:
                    trace, _ = haiku(ATTEMPT.format(ctx=retrieve(q), q=q))
                    records.append({"src": src, "task": rec.task, "subq": q,
                                    "trace": trace, "label": verify(q, trace)})
            except Exception as e:                            # noqa: BLE001
                print(f"  {src}[{ti}] failed: {e!r}", flush=True); continue
            print(f"  {src} task {ti+1}/{len(recs)}: {len(subqs)} components", flush=True)

    out = os.path.join(HERE, "results", "component_records.json")
    json.dump(records, open(out, "w"))
    print(f"\nsaved {len(records)} component records -> {out}", flush=True)

    # forecast per component
    V = [v.tolist() for v in embed([r["trace"] for r in records])]
    lab = [r["label"] for r in records]
    src = [r["src"] for r in records]
    print("\n" + "=" * 56)
    print(f"  PER-COMPONENT FORECAST  (n={len(records)} components)")
    print("=" * 56)
    for s in sorted(set(src)):
        idx = [i for i, x in enumerate(src) if x == s]
        sub = [lab[i] for i in idx]
        if len(set(sub)) < 2:
            print(f"  [{s}] n={len(idx)} success={sum(sub)} — one class, skip"); continue
        a = auc(sub, knn_predict([V[i] for i in idx], sub, 10))
        print(f"  [{s} within]   n={len(idx)}  success={sum(sub)}/{len(sub)}  AUC={a:.3f}")
    # leave-one-task-type-out at the COMPONENT level
    for test_s in sorted(set(src)):
        te = [i for i, x in enumerate(src) if x == test_s]
        tr = [i for i, x in enumerate(src) if x != test_s]
        if len(set(lab[i] for i in te)) < 2 or not tr:
            continue
        a = auc([lab[i] for i in te],
                knn_predict_cross([V[i] for i in tr], [lab[i] for i in tr], [V[i] for i in te], 10))
        print(f"  [{test_s} <- others]  transfer AUC={a:.3f}")
    print("=" * 56)
    print("READ: component AUC >> the task-level fan-out ~chance (0.45) means decomposing")
    print("into checkable units made failure predictable — generically, any task.")


if __name__ == "__main__":
    main()
