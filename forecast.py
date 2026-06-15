"""Core, dependency-light primitives for the failure-forecasting MVP.

The one question this answers: do *trace embeddings* carry outcome signal? i.e. can a
trajectory's nearest neighbours (in embedding space) predict whether it succeeds, better
than chance and better than knowing the task alone? Pure functions so they can be unit-
tested offline; the real run lives in eval/mvp_failure_forecast.py.
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


def auc(labels: List[int], scores: List[float]) -> float:
    """ROC-AUC via Mann-Whitney: P(score_pos > score_neg), ties = 0.5. 0.5 == chance.
    Returns nan if a class is missing."""
    pos = [s for l, s in zip(labels, scores) if l == 1]
    neg = [s for l, s in zip(labels, scores) if l == 0]
    if not pos or not neg:
        return float("nan")
    wins = sum(1.0 if p > q else 0.5 if p == q else 0.0 for p in pos for q in neg)
    return wins / (len(pos) * len(neg))
