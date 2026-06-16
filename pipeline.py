"""trace_use — the generic, importable surface for trace-based failure forecasting.

Forecast (and pre-empt) agent failure from execution traces, on ANY task. The pipeline is
`decompose -> attempt -> verify -> forecast`; the ONLY task-specific input is a VERIFIER
(a `(question, answer) -> 0..1` check — a unit test, an LLM judge, a regex). Everything
else is task-agnostic. Built on the dependency-light primitives in `forecast.py`.

    from pipeline import Forecaster, decompose, attempt, gold_judge
    fc = Forecaster(embedder).fit(traces, labels)
    if fc.should_intervene(new_trace):     # spend a retry/verify only when likely to fail
        ...
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Callable, List, Optional, Protocol, Sequence, Tuple

from forecast import knn_predict_cross

Agent = Callable[[str], object]            # prompt -> (text, tokens) | text

# ── prompts for the generic operations (no dataset logic) ────────────────────────
_DECOMPOSE = ("Break the task into the minimal list of ATOMIC, independently-checkable "
              "sub-questions whose answers together fully answer it. One sub-question per "
              "line, no numbering, no commentary.\n\nTask: {task}")
_ATTEMPT = ("{ctx}\n\nQuestion: {q}\n\nReason step by step, noting what you look for and "
            "what you find, then end with 'ANSWER: ...'.")
_JUDGE = ("Gold answer to the overall task:\n{gold}\n\nSub-question: {q}\nProposed answer: "
          "{a}\n\nIs the proposed answer correct and supported by the gold? Reply YES or NO.")


def _text(out: object) -> str:
    return str(out[0]) if isinstance(out, tuple) else str(out)


# ── generic task operations ──────────────────────────────────────────────────────
def decompose(task: str, agent: Agent, cap: int = 8) -> List[str]:
    """Split any task into atomic, independently-checkable sub-questions."""
    lines = _text(agent(_DECOMPOSE.format(task=task))).splitlines()
    qs = [ln.strip(" -*\t").strip() for ln in lines if len(ln.strip()) > 6]
    return qs[:cap] or [task]


def attempt(question: str, context: str, agent: Agent) -> str:
    """Attempt one (sub-)question over `context`; returns the reasoning trace + answer."""
    return _text(agent(_ATTEMPT.format(ctx=context, q=question)))


class Verifier(Protocol):
    def __call__(self, question: str, answer: str) -> float: ...


def gold_judge(gold: str, agent: Agent) -> Verifier:
    """Verifier against a known gold answer (an LLM judge). Use when you have ground truth."""
    def verify(question: str, answer: str) -> float:
        out = _text(agent(_JUDGE.format(gold=str(gold)[:4000], q=question, a=answer[:1500])))
        return 1.0 if "yes" in out.strip().lower()[:5] else 0.0
    return verify


# ── zero-labeling auto-verifiers (no gold, no human) ─────────────────────────────
_SELF_JUDGE = ("{ev}Question: {q}\nProposed answer: {a}\n\nJudge strictly: is the proposed "
               "answer correct and well-supported? Reply YES or NO.")


def self_judge(judge_agent: Agent, evidence_fn: Optional[Callable[[str], str]] = None) -> Verifier:
    """Reference-free auto-verifier — NO gold answer needed. A model grades whether the
    answer is correct/supported (by retrieved evidence, if `evidence_fn` is given). Fully
    automatic. NOTE: use an INDEPENDENT (ideally stronger/cheaper-but-different) judge model
    than the one being judged — a model grading itself is overconfident and learns the judge,
    not the truth."""
    def verify(question: str, answer: str) -> float:
        ev = f"Evidence:\n{evidence_fn(question)[:3000]}\n\n" if evidence_fn else ""
        out = _text(judge_agent(_SELF_JUDGE.format(ev=ev, q=question, a=answer[:1500])))
        return 1.0 if "yes" in out.strip().lower()[:5] else 0.0
    return verify


def _final_answer(text: str) -> str:
    m = re.search(r"answer\s*:\s*(.+)", text, re.I)
    return (m.group(1) if m else text).strip().lower()


def self_consistency(resample: Callable[[str], str], samples: int = 3) -> Verifier:
    """Label-free AND judge-free auto-verifier: re-attempt the question `samples` times and
    return the fraction of independent runs whose final answer matches the given one. High
    agreement => consistent => likely correct; low => flaky => likely failure. Best when the
    final answer is short/extractable (a number, an entity)."""
    def verify(question: str, answer: str) -> float:
        target = _final_answer(answer)
        finals = [_final_answer(resample(question)) for _ in range(max(1, samples))]
        return sum(1 for f in finals if f == target) / len(finals)
    return verify


def tiered_judge(
    fast_agent: Agent,
    strong_agent: Agent,
    gold: str,
    uncertainty_band: Tuple[float, float] = (0.35, 0.65),
) -> Verifier:
    """Two-tier verifier: fast_agent (e.g. Haiku) judges first. Only calls strong_agent
    (e.g. Opus) when the fast verdict falls in the uncertainty band — typically ~20-30% of
    cases. Returns the strong verdict on escalation, fast verdict otherwise."""
    fast_v   = gold_judge(gold, fast_agent)
    strong_v = gold_judge(gold, strong_agent)
    lo, hi   = uncertainty_band

    def verify(question: str, answer: str) -> float:
        score = fast_v(question, answer)
        if lo <= score <= hi:
            return strong_v(question, answer)
        return score
    return verify


def make_retriever(chunks: Sequence[str], embedder) -> Callable[..., str]:
    """Embedding retriever over a corpus: returns the top chunks (by similarity to a query)
    up to a word budget. Embeds the corpus once."""
    import numpy as np
    chunks = [c for c in chunks if c.strip()]
    cv = np.asarray(embedder(chunks), dtype="float32") if chunks else None

    def retrieve(query: str, words: int = 1200) -> str:
        if cv is None:
            return ""
        qv = np.asarray(embedder([query]), dtype="float32")[0]
        order = np.argsort(-(cv @ qv))
        picked, used = [], 0
        for j in order:
            w = len(chunks[j].split())
            if picked and used + w > words:
                break
            picked.append(chunks[j]); used += w
        return "\n\n".join(picked)
    return retrieve


# ── the forecaster (non-parametric, grows online) ────────────────────────────────
class Forecaster:
    """k-NN failure forecaster over a trace store. Embed a trace, retrieve the most similar
    past traces, predict P(fail) from their outcomes. No training; `add()` new traces as
    they finish so the store improves with use."""

    def __init__(self, embedder, k: int = 10):
        self.embedder = embedder
        self.k = k
        self._vecs: List[list] = []
        self._labels: List[int] = []

    def fit(self, traces: Sequence[str], labels: Sequence[int]) -> "Forecaster":
        self._vecs = [v.tolist() for v in self.embedder(list(traces))]
        self._labels = [int(x) for x in labels]
        return self

    def add(self, trace: str, label: int) -> None:
        self._vecs.append(self._embed1(trace))
        self._labels.append(int(label))

    def predict_fail(self, trace: str) -> float:
        """Probability this trace's component FAILS (0..1)."""
        if len(set(self._labels)) < 2:
            base = sum(self._labels) / max(1, len(self._labels))
            return 1.0 - base
        succ = knn_predict_cross(self._vecs, self._labels, [self._embed1(trace)], self.k)[0]
        return 1.0 - succ

    def should_intervene(self, trace: str, threshold: float = 0.5) -> bool:
        """Spend a retry/verify/escalation only when failure is likely."""
        return self.predict_fail(trace) >= threshold

    def _embed1(self, trace: str) -> list:
        return self.embedder([trace])[0].tolist()
