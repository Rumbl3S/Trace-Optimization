"""Per-component forecasting — the headline experiment (and the recommended usage).

Predict failure for each CHECKABLE COMPONENT of a task, not the whole task. Uses the
generic `pipeline` surface end to end (`decompose -> attempt -> verify`), so the same code
runs on any task — only the verifier is task-specific (here a gold judge). Reports
within-dataset and cross-task (leave-one-type-out) component-level AUC.

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

import llm
import run_fanoutqa as fq
import run_musique as mu
from demo_embed_compare import haiku, _build_openai
from pipeline import decompose, attempt, gold_judge, make_retriever
from forecast import knn_predict, knn_predict_cross, auc

llm._ensure_api_key()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", type=int, default=18)
    ap.add_argument("--max-components", type=int, default=6)
    args = ap.parse_args()
    embed = _build_openai()

    records = []
    for src, recs in [("fanout", fq.load(args.tasks)), ("musique", mu.load(args.tasks))]:
        for ti, rec in enumerate(recs):
            try:
                retrieve = make_retriever(rec.context_chunks, embed)
                verify = gold_judge(rec.gold, haiku)
                subqs = decompose(rec.task, haiku, args.max_components)
                for q in subqs:
                    trace = attempt(q, retrieve(q), haiku)
                    records.append({"src": src, "task": rec.task, "subq": q,
                                    "trace": trace, "label": int(verify(q, trace))})
            except Exception as e:                            # noqa: BLE001
                print(f"  {src}[{ti}] failed: {e!r}", flush=True); continue
            print(f"  {src} task {ti+1}/{len(recs)}: {len(subqs)} components", flush=True)

    out = os.path.join(HERE, "results", "component_records.json")
    json.dump(records, open(out, "w"))
    print(f"\nsaved {len(records)} component records -> {out}", flush=True)

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
            print(f"  [{s} within] n={len(idx)} success={sum(sub)} — one class, skip"); continue
        a = auc(sub, knn_predict([V[i] for i in idx], sub, 10))
        print(f"  [{s} within]   n={len(idx)}  success={sum(sub)}/{len(sub)}  AUC={a:.3f}")
    for test_s in sorted(set(src)):
        te = [i for i, x in enumerate(src) if x == test_s]
        tr = [i for i, x in enumerate(src) if x != test_s]
        if len(set(lab[i] for i in te)) < 2 or not tr:
            continue
        a = auc([lab[i] for i in te],
                knn_predict_cross([V[i] for i in tr], [lab[i] for i in tr], [V[i] for i in te], 10))
        print(f"  [{test_s} <- others]  transfer AUC={a:.3f}")
    print("=" * 56)


if __name__ == "__main__":
    main()
