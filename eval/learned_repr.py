"""PART 3 — does a LEARNED representation beat off-the-shelf embeddings?

Baseline = k-NN on raw OpenAI trace embeddings (non-parametric, ~0.85 on MuSiQue).
Learned (all leave-one-out cross-validated, so no leakage):
  - PCA + k-NN      (unsupervised dimensionality reduction)
  - logistic probe  (supervised linear classifier on PCA features)
  - LDA projection  (supervised 1-D direction that separates success/failure) + threshold

If a learned variant clears the raw-kNN baseline by more than noise, representation
learning pays off at this data scale. If not, the honest read is 'off-the-shelf is
already as good as it gets here — the bottleneck is DATA, not method.'

  python eval/learned_repr.py results/trajectories_balanced.json
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(__file__)
RS = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path[:0] = [RS, os.path.join(RS, "eval"), os.path.abspath(os.path.join(HERE, ".."))]

import numpy as np
from demo_embed_compare import _build_openai
from forecast import knn_predict, auc


def standardize(X):
    mu, sd = X.mean(0), X.std(0) + 1e-8
    return (X - mu) / sd


def pca_project(X, d):
    Xc = X - X.mean(0)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc @ Vt[:d].T


def logreg_fit(X, y, l2=1.0, iters=400, lr=0.2):
    n, d = X.shape
    w, b = np.zeros(d), 0.0
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-(X @ w + b)))
        g = p - y
        w -= lr * (X.T @ g / n + l2 * w / n)
        b -= lr * g.mean()
    return w, b


def lda_dir(X, y):
    X1, X0 = X[y == 1], X[y == 0]
    Sw = (np.cov(X1.T) * len(X1) + np.cov(X0.T) * len(X0)) + np.eye(X.shape[1]) * 1e-1
    return np.linalg.solve(Sw, X1.mean(0) - X0.mean(0))


def loo(X, y, fit, pred):
    n = len(y)
    out = np.zeros(n)
    for i in range(n):
        idx = [j for j in range(n) if j != i]
        m = fit(X[np.array(idx)], y[np.array(idx)])
        out[i] = pred(X[i], m)
    return out


def evaluate(Xraw, y, d=15):
    y = np.asarray(y, dtype="float64")
    Z = pca_project(standardize(Xraw), min(d, Xraw.shape[0] - 2, Xraw.shape[1]))
    res = {}
    res["raw kNN@10"] = auc(list(y.astype(int)), knn_predict([v.tolist() for v in Xraw], list(y.astype(int)), 10))
    res["PCA kNN@10"] = auc(list(y.astype(int)), knn_predict([v.tolist() for v in Z], list(y.astype(int)), 10))
    lr = loo(Z, y, lambda Xt, yt: logreg_fit(Xt, yt),
             lambda x, m: 1.0 / (1.0 + np.exp(-(x @ m[0] + m[1]))))
    res["logreg(PCA)"] = auc(list(y.astype(int)), list(lr))
    ld = loo(Z, y, lambda Xt, yt: lda_dir(Xt, yt), lambda x, w: float(x @ w))
    res["LDA(PCA)"] = auc(list(y.astype(int)), list(ld))
    return res


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "results", "trajectories_balanced.json")
    recs = json.load(open(path))
    embed = _build_openai()
    X = np.asarray(embed([r["trace"] for r in recs]), dtype="float64")
    y = [r["label"] for r in recs]
    srcs = [r["src"] for r in recs]

    print("=" * 52)
    print(f"  LEARNED REPRESENTATION TEST  ({os.path.basename(path)})")
    print("=" * 52)
    for title, idx in [("ALL", list(range(len(recs))))] + \
                      [(f"{s} ONLY", [i for i, x in enumerate(srcs) if x == s]) for s in sorted(set(srcs))]:
        sub_y = [y[i] for i in idx]
        if len(set(sub_y)) < 2 or len(idx) < 12:
            print(f"\n  [{title}] n={len(idx)} — too small / one class, skip"); continue
        res = evaluate(X[np.array(idx)], sub_y)
        print(f"\n  [{title}]  n={len(idx)}  success={sum(sub_y)}/{len(sub_y)}")
        for name, a in res.items():
            print(f"    {name:<14}{a:>8.3f}")
    print("\n" + "=" * 52)
    print("READ: learned beats 'raw kNN@10' by > ~0.03 -> representation learning")
    print("helps. Otherwise off-the-shelf embeddings are the ceiling at this n.")


if __name__ == "__main__":
    main()
