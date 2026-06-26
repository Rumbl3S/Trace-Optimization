"""Core, dependency-light primitives for the failure-forecasting MVP.

The one question this answers: do *trace embeddings* carry outcome signal? i.e. can a
trajectory's nearest neighbours (in embedding space) predict whether it succeeds, better
than chance and better than knowing the task alone? Pure functions so they can be unit-
tested offline; the experiments that use them live in eval/.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence


@dataclass
class TrajectoryRecord:
    task: str
    src: str                      # 'fanout' | 'musique'
    trace: str                    # the agent's reasoning/exploration text (the trajectory)
    answer: str
    score: float                  # continuous metric (loose / contains)
    label: int                    # 1 = success, 0 = failure (binarised score)
    task_vec: Optional[list] = None
    trace_vec: Optional[list] = None


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb + 1e-12)


def knn_predict(vecs: List[Sequence[float]], labels: List[int], k: int) -> List[float]:
    """LEAVE-ONE-OUT: predicted success for i = mean label of its k nearest neighbours
    (cosine, excluding itself). No training — this IS the non-parametric forecaster."""
    n = len(vecs)
    preds = []
    for i in range(n):
        sims = sorted(((cosine(vecs[i], vecs[j]), labels[j]) for j in range(n) if j != i),
                      key=lambda t: t[0], reverse=True)
        top = sims[:k] or [(0.0, sum(labels) / max(1, n))]
        preds.append(sum(l for _, l in top) / len(top))
    return preds


def knn_predict_cross(train_vecs, train_vals, test_vecs, k: int) -> List[float]:
    """Predict each TEST point from its k nearest TRAIN neighbours (cosine). Used for
    leave-one-task-type-out: train on some task families, predict an unseen one."""
    preds = []
    for tv in test_vecs:
        sims = sorted(((cosine(tv, trv), v) for trv, v in zip(train_vecs, train_vals)),
                      key=lambda t: t[0], reverse=True)[:k]
        preds.append(sum(v for _, v in sims) / max(1, len(sims)))
    return preds


def _ranks(xs: Sequence[float]) -> List[float]:
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2.0
        for t in range(i, j + 1):
            ranks[order[t]] = avg
        i = j + 1
    return ranks


def spearman(a: Sequence[float], b: Sequence[float]) -> float:
    """Rank correlation — does predicted *degree of success* track the actual score?
    Robust where a binary AUC is uninformative (e.g. partial-coverage fan-out)."""
    ra, rb = _ranks(a), _ranks(b)
    n = len(a)
    ma, mb = sum(ra) / n, sum(rb) / n
    cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    va = math.sqrt(sum((x - ma) ** 2 for x in ra))
    vb = math.sqrt(sum((y - mb) ** 2 for y in rb))
    return cov / (va * vb + 1e-12)


def auc(labels: List[int], scores: List[float]) -> float:
    """ROC-AUC via Mann-Whitney: P(score_pos > score_neg), ties = 0.5. 0.5 == chance.
    Returns nan if a class is missing."""
    pos = [s for l, s in zip(labels, scores) if l == 1]
    neg = [s for l, s in zip(labels, scores) if l == 0]
    if not pos or not neg:
        return float("nan")
    wins = sum(1.0 if p > q else 0.5 if p == q else 0.0 for p in pos for q in neg)
    return wins / (len(pos) * len(neg))
