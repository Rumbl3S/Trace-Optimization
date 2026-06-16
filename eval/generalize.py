"""Make the forecaster GENERALIZABLE — two tests, no new data.

A) CONTINUOUS target (per-component proxy): instead of a binary pass/fail at 0.5, predict
   the actual coverage SCORE (degree of success) from trace neighbours. Reports Spearman
   (does predicted degree track actual?) next to binary AUC. The bet: fan-out's signal is
   real but hidden by the 0.5 cutoff — continuous prediction should surface it.

B) LEAVE-ONE-TASK-TYPE-OUT: train the store on one task family, predict an UNSEEN one
   (musique<-fanout and fanout<-musique). This is the real 'works for anything' test —
   does the forecaster transfer across task types it wasn't built on?

  python eval/generalize.py eval/results/trajectories_balanced.json
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))

from agents import _build_openai
from forecast import knn_predict, knn_predict_cross, auc, spearman


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "results", "trajectories_balanced.json")
    recs = json.load(open(path))
    embed = _build_openai()
    V = [v.tolist() for v in embed([r["trace"] for r in recs])]
    score = [float(r["score"]) for r in recs]
    label = [int(r["label"]) for r in recs]
    src = [r["src"] for r in recs]
    srcs = sorted(set(src))

    print("=" * 60)
    print(f"  GENERALIZABILITY  ({os.path.basename(path)}, n={len(recs)})")
    print("=" * 60)

    # A) continuous (degree-of-success) vs binary, within each task family
    print("\n  [A] CONTINUOUS target vs binary (within-dataset, k=10)")
    print(f"  {'dataset':<10}{'binary AUC':>12}{'cont. Spearman':>17}")
    for s in srcs:
        idx = [i for i, x in enumerate(src) if x == s]
        if len(idx) < 12:
            continue
        v = [V[i] for i in idx]
        bin_auc = auc([label[i] for i in idx], knn_predict(v, [label[i] for i in idx], 10))
        rho = spearman([score[i] for i in idx], knn_predict(v, [score[i] for i in idx], 10))
        print(f"  {s:<10}{bin_auc:>12.3f}{rho:>17.3f}")
    print("       (Spearman > 0 means degree-of-success IS predictable even when the")
    print("        binary 0.5 cutoff looks like chance — the per-component fix.)")

    # B) leave-one-task-type-out transfer
    print("\n  [B] LEAVE-ONE-TASK-TYPE-OUT (train on others, predict unseen type, k=10)")
    print(f"  {'predict':<10}{'from':<12}{'transfer AUC':>13}{'Spearman':>11}")
    for test_s in srcs:
        te = [i for i, x in enumerate(src) if x == test_s]
        tr = [i for i, x in enumerate(src) if x != test_s]
        if len(set(label[i] for i in te)) < 2 or len(tr) < 5:
            print(f"  {test_s:<10}{'others':<12}{'— one class / too few':>13}"); continue
        trv, trl, trsc = [V[i] for i in tr], [label[i] for i in tr], [score[i] for i in tr]
        tev = [V[i] for i in te]
        a = auc([label[i] for i in te], knn_predict_cross(trv, trl, tev, 10))
        r = spearman([score[i] for i in te], knn_predict_cross(trv, trsc, tev, 10))
        print(f"  {test_s:<10}{'others':<12}{a:>13.3f}{r:>11.3f}")
    print("       (AUC > 0.5 / Spearman > 0 across types => it generalizes to unseen tasks.)")
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
