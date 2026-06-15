"""THE PRODUCT LOOP — turn forecasts into saved tokens/latency.

A forecaster is only useful if it lets you spend expensive work (a retry, a verify, an
escalation) ONLY where it's needed. This measures exactly that: rank components by
predicted-fail probability (leave-one-out, from the trace store), 'intervene' on the top
B% of the budget, and report how many of the REAL failures you catch vs spending the same
budget at random.

Good forecaster => failures concentrate at the top => you catch most of them while
touching a fraction of components => tokens/latency saved at ~equal failure coverage.

  python eval/intervention_policy.py eval/results/component_records.json
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(__file__)
RS = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path[:0] = [RS, os.path.join(RS, "eval"), os.path.abspath(os.path.join(HERE, ".."))]

from demo_embed_compare import _build_openai
from forecast import knn_predict, auc


def gain(fail_prob, labels, budgets=(0.1, 0.2, 0.3, 0.4, 0.5)):
    n = len(labels)
    order = sorted(range(n), key=lambda i: fail_prob[i], reverse=True)
    total_fail = sum(1 for l in labels if l == 0)
    rows = []
    for b in budgets:
        k = max(1, int(round(b * n)))
        caught = sum(1 for i in order[:k] if labels[i] == 0)
        recall = caught / max(1, total_fail)
        rows.append((b, recall, recall / b))           # budget, failure-recall, lift vs random
    # budget needed to catch 80% of failures
    need = 1.0
    cum = 0
    for rank, i in enumerate(order, 1):
        cum += (labels[i] == 0)
        if cum >= 0.8 * total_fail:
            need = rank / n
            break
    return rows, need


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "results", "component_records.json")
    recs = json.load(open(path))
    embed = _build_openai()
    V = [v.tolist() for v in embed([r["trace"] for r in recs])]
    lab = [r["label"] for r in recs]
    src = [r["src"] for r in recs]

    def report(title, idx):
        sub = [lab[i] for i in idx]
        if len(set(sub)) < 2:
            print(f"\n  [{title}] one class, skip"); return
        vv = [V[i] for i in idx]
        succ = knn_predict(vv, sub, 10)                # LOO success prob
        fail_prob = [1 - p for p in succ]
        a = auc(sub, succ)
        rows, need = gain(fail_prob, sub)
        print(f"\n  [{title}]  n={len(idx)}  failures={sum(1 for l in sub if l==0)}  AUC={a:.3f}")
        print(f"  {'budget':>8}{'failures caught':>18}{'vs random':>11}")
        for b, recall, lift in rows:
            print(f"  {int(b*100):>6}% {recall*100:>14.0f}% {lift:>10.2f}x")
        print(f"  -> catch 80% of failures by intervening on only {need*100:.0f}% of components"
              f"  (random would need 80%)")

    print("=" * 58)
    print(f"  INTERVENTION POLICY — gain from forecast-gating  (n={len(recs)})")
    print("=" * 58)
    report("ALL components", list(range(len(recs))))
    for s in sorted(set(src)):
        report(f"{s} only", [i for i, x in enumerate(src) if x == s])
    print("\n" + "=" * 58)
    print("READ: 'failures caught' at a small budget >> the budget % means the forecast")
    print("lets you spend fix/verify/retry tokens only where they matter — the savings.")


if __name__ == "__main__":
    main()
