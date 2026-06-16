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
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Protocol, Sequence, Tuple

from forecast import knn_predict_cross, cosine

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
        tail = answer[-3000:] if len(answer) > 3000 else answer
        out = _text(agent(_JUDGE.format(gold=str(gold)[:4000], q=question, a=tail)))
        return 1.0 if "yes" in out.strip().lower()[:5] else 0.0
    return verify


# ── zero-labeling auto-verifiers (no gold, no human) ─────────────────────────────
_SELF_JUDGE = ("{ev}Question: {q}\n\nAgent response:\n{a}\n\n"
               "Does this response adequately answer the question? "
               "Reply YES if the answer is reasonable and addresses what was asked "
               "(even if approximate or sourced from the agent's knowledge). "
               "Reply NO only if the answer is clearly wrong, refuses to engage, "
               "or completely fails to address the question. Reply YES or NO.")


def self_judge(judge_agent: Agent, evidence_fn: Optional[Callable[[str], str]] = None) -> Verifier:
    """Reference-free auto-verifier — NO gold answer needed. A model grades whether the
    answer is correct/supported (by retrieved evidence, if `evidence_fn` is given). Fully
    automatic. NOTE: use an INDEPENDENT (ideally stronger/cheaper-but-different) judge model
    than the one being judged — a model grading itself is overconfident and learns the judge,
    not the truth."""
    def verify(question: str, answer: str) -> float:
        ev = f"Evidence:\n{evidence_fn(question)[:3000]}\n\n" if evidence_fn else ""
        # use the tail of the trace — the conclusion and final answer live there,
        # not in the first 1500 chars which are typically raw tool-call output
        tail = answer[-3000:] if len(answer) > 3000 else answer
        out = _text(judge_agent(_SELF_JUDGE.format(ev=ev, q=question, a=tail)))
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


# ── result container ─────────────────────────────────────────────────────────
@dataclass
class ComponentResult:
    question:  str
    trace:     str
    label:     int
    p_fail:    Optional[float]
    retried:   bool
    neighbor:  Optional[str]   # nearest stored failure trace excerpt (for explain)


@dataclass
class TaskResult:
    task:       str
    components: List[ComponentResult] = field(default_factory=list)

    @property
    def n_pass(self):       return sum(1 for c in self.components if c.label == 1)
    @property
    def n_fail(self):       return sum(1 for c in self.components if c.label == 0)
    @property
    def n_intervened(self): return sum(1 for c in self.components if c.retried)

    def summary(self) -> str:
        lines = [f"Task: {self.task}",
                 f"Components: {len(self.components)}  Pass: {self.n_pass}  "
                 f"Fail: {self.n_fail}  Interventions: {self.n_intervened}"]
        for i, c in enumerate(self.components):
            pf = f"P(fail)={c.p_fail:.2f}" if c.p_fail is not None else "no forecast"
            outcome = "PASS" if c.label == 1 else "FAIL"
            lines.append(f"  [{i+1}] {outcome}  {pf}  {c.question[:60]}")
        return "\n".join(lines)


# ── the forecaster (non-parametric, grows online) ─────────────────────────────
class Forecaster:
    """k-NN failure forecaster over a trace store. Embed a trace, retrieve the most similar
    past traces, predict P(fail) from their outcomes. No training; `add()` new traces as
    they finish so the store improves with use."""

    def __init__(self, embedder, k: int = 10):
        self.embedder = embedder
        self.k = k
        self._vecs:   List[list] = []
        self._labels: List[int]  = []
        self._traces: List[str]  = []   # raw traces stored for explain()

    def fit(self, traces: Sequence[str], labels: Sequence[int]) -> "Forecaster":
        self._traces = list(traces)
        self._vecs   = [v.tolist() for v in self.embedder(list(traces))]
        self._labels = [int(x) for x in labels]
        return self

    def add(self, trace: str, label: int) -> None:
        self._traces.append(trace)
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

    def explain(self, trace: str, k: int = 3) -> List[Dict]:
        """Return the k nearest stored traces driving this prediction.

        Each entry: {"similarity": float, "label": int, "excerpt": str}
        Use to understand WHY a component was flagged.
        """
        if not self._vecs:
            return []
        vec = self._embed1(trace)
        ranked = sorted(
            ((cosine(vec, v), l, t)
             for v, l, t in zip(self._vecs, self._labels, self._traces)),
            reverse=True,
        )[:k]
        return [{"similarity": round(s, 3),
                 "label": l,
                 "outcome": "pass" if l == 1 else "fail",
                 "excerpt": t[:200]}
                for s, l, t in ranked]

    def nearest_failure(self, trace: str) -> Optional[str]:
        """Return excerpt of the most similar stored failure, or None."""
        neighbors = self.explain(trace, k=5)
        for n in neighbors:
            if n["label"] == 0:
                return n["excerpt"]
        return None

    def _embed1(self, trace: str) -> list:
        return self.embedder([trace])[0].tolist()


# ── high-level orchestrator ───────────────────────────────────────────────────
def run_task(
    task:        str,
    agent:       Agent,
    verifier:    Optional[Verifier] = None,
    forecaster:  Optional[Forecaster] = None,
    retriever:   Optional[Callable[..., str]] = None,
    threshold:   float = 0.5,
    cap:         int = 8,
    display:     bool = True,
    retry:       bool = True,
) -> TaskResult:
    """Run the full trace_use pipeline on any task.

    decompose → attempt → [forecast] → [intervene+retry] → [verify] → store

    Args:
        task:       Any natural-language task.
        agent:      Callable (prompt -> text | (text, tokens)). Use haiku, opus,
                    or tool_agent() from agents.py.
        verifier:   Optional (question, answer) -> 0..1 scorer. If omitted,
                    self_judge is used when a forecaster is present.
        forecaster: Fitted or empty Forecaster. If None, no failure prediction
                    is run — useful for bootstrapping a trace store.
        retriever:  Optional retrieval fn (query -> context string).
        threshold:  P(fail) threshold to trigger intervention (default 0.5).
        cap:        Max sub-questions to decompose into (default 8).
        display:    Show the live Rich terminal display (default True).
        retry:      Retry once on predicted failure (default True).

    Returns:
        TaskResult with all component traces, labels, predictions, and neighbors.
    """
    from display import TraceDisplay, print_summary

    store_n    = len(forecaster._vecs) if forecaster else 0
    agent_name = getattr(agent, "__name__", "agent")
    result     = TaskResult(task=task)

    disp = TraceDisplay(task, agent_name=agent_name, store_size=store_n)

    with disp:
        sub_qs = decompose(task, agent, cap=cap)
        disp.set_components(sub_qs)

        for i, q in enumerate(sub_qs):
            disp.set_attempting(i)
            ctx   = retriever(q) if retriever else ""
            trace = attempt(q, ctx, agent)

            p_fail   = None
            retried  = False
            neighbor = None

            if forecaster and len(forecaster._vecs) >= 2:
                p_fail   = forecaster.predict_fail(trace)
                neighbor = forecaster.nearest_failure(trace)

                if retry and p_fail >= threshold:
                    # retry once with the same question
                    trace   = attempt(q, ctx, agent)
                    retried = True
                    p_fail  = forecaster.predict_fail(trace)

            # verify / label
            if verifier:
                label = int(verifier(q, trace) >= 0.5)
            else:
                label = 1   # optimistic default when no verifier

            disp.set_result(i, p_fail if p_fail is not None else 0.0,
                            label, retried=retried, neighbor=neighbor)

            if forecaster:
                forecaster.add(trace, label)
                disp.update_store(len(forecaster._vecs))

            result.components.append(ComponentResult(
                question=q, trace=trace, label=label,
                p_fail=p_fail, retried=retried, neighbor=neighbor,
            ))

    if display:
        print_summary([
            {"question": c.question, "p_fail": c.p_fail,
             "label": c.label, "retried": c.retried}
            for c in result.components
        ])

    return result
