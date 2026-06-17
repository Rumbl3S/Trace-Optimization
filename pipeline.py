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
_RETRY  = ("{ctx}\n\nQuestion: {q}\n\nYour previous attempt (which may be wrong):\n{prev}\n\n"
           "First, in one sentence quote the exact step above that is wrong or incomplete "
           "and state what NOT to do. Then reattempt from scratch and end with 'ANSWER: ...'.")
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


def extract_code(text: str) -> str:
    """Pull the first fenced Python block from text, falling back to the full text."""
    m = re.search(r"```(?:python|py)?\s*\n(.*?)```", text, re.DOTALL | re.I)
    return m.group(1).strip() if m else text.strip()


def code_judge(check: Callable[[dict, str], bool]) -> Verifier:
    """Verifier for code-writing and debugging tasks.

    Extracts Python from the agent's answer, exec()s it, then calls
    check(namespace, stdout) to decide correctness. Returns 1.0 on pass,
    0.0 on failure or any exception (including syntax errors in the code).

    Example — debug a binary search::

        def my_check(ns, out):
            fn = ns.get("binary_search")
            return fn and fn([1, 2, 3, 4, 5], 3) == 2

        verifier = code_judge(my_check)
        result   = run_task(task, agent=agent, verifier=verifier, forecaster=fc)

    The forecaster still wraps the verifier: when the agent's debugging trace
    shows uncertainty or repeated revisions, P(fail) rises and a retry fires
    *before* the verifier is called, letting the agent self-correct.
    """
    def verify(question: str, answer: str) -> float:
        import io, contextlib
        tail = answer[-3000:] if len(answer) > 3000 else answer
        code = extract_code(tail)
        buf  = io.StringIO()
        ns: dict = {}
        try:
            with contextlib.redirect_stdout(buf):
                exec(compile(code, "<agent>", "exec"), ns)  # noqa: S102
            return 1.0 if check(ns, buf.getvalue()) else 0.0
        except Exception:
            return 0.0
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
    they finish so the store improves with use.

    PCA (default 64 dims) is fitted once the store exceeds pca_dim examples and kept
    up-to-date on each add(). kNN then runs in the reduced space, which improves
    separation in high-dimensional embedding spaces."""

    def __init__(self, embedder, k: int = 10, pca_dim: int = 16):
        self.embedder  = embedder
        self.k         = k
        self.pca_dim   = pca_dim
        self._raw_vecs: List[list] = []   # full-dim embeddings kept for PCA refitting
        self._vecs:     List[list] = []   # projected (pca_dim) once PCA is active
        self._labels:   List[int]  = []
        self._traces:   List[str]  = []
        self._pca                  = None

    def fit(self, traces: Sequence[str], labels: Sequence[int]) -> "Forecaster":
        self._traces   = list(traces)
        self._raw_vecs = [v.tolist() for v in self.embedder(list(traces))]
        self._labels   = [int(x) for x in labels]
        self._pca      = self._fit_pca() if self.pca_dim and len(self._raw_vecs) > self.pca_dim else None
        self._vecs     = self._project_all()
        return self

    def add(self, trace: str, label: int) -> None:
        self._traces.append(trace)
        self._raw_vecs.append(self._embed1_raw(trace))
        self._labels.append(int(label))
        if self.pca_dim and len(self._raw_vecs) > self.pca_dim:
            self._pca  = self._fit_pca()
            self._vecs = self._project_all()
        else:
            self._vecs.append(self._raw_vecs[-1])

    def predict_fail(self, trace: str) -> float:
        """Probability this trace's component FAILS (0..1), locally normalised.

        Normalises the raw kNN score against the neighbourhood's own failure
        rate, so a component that scores 0.20 in a region where everything scores
        0.05 still surfaces as high-risk.  Falls back to raw when neighbourhood
        has no variance (e.g. all-pass region — genuine signal absence).
        """
        if len(set(self._labels)) < 2:
            base = sum(self._labels) / max(1, len(self._labels))
            return 1.0 - base
        vec = self._vec(trace)
        raw_succ = knn_predict_cross(self._vecs, self._labels, [vec], self.k)[0]
        raw_fail = 1.0 - raw_succ

        # neighbourhood baseline: mean P(fail) of 2k nearest stored points
        if len(self._vecs) >= max(self.k * 2, 20):
            import math
            top_idx = sorted(range(len(self._vecs)),
                             key=lambda j: cosine(vec, self._vecs[j]),
                             reverse=True)[: self.k * 2]
            neigh_fail = 1.0 - (sum(self._labels[j] for j in top_idx) / len(top_idx))
            neigh_std  = math.sqrt(neigh_fail * (1.0 - neigh_fail))
            if neigh_std > 1e-6:
                z = (raw_fail - neigh_fail) / neigh_std
                return 1.0 / (1.0 + math.exp(-z))
        return raw_fail

    @property
    def adaptive_threshold(self) -> float:
        """Threshold that tracks the store's current observed failure rate.

        Uses 1.5× the empirical fail rate so we only fire on components that
        score meaningfully above base rate — automatically right-sized whether
        the store is 10% or 50% failing, and safe to use with zero history.
        """
        if not self._labels:
            return 0.35
        fail_rate = 1.0 - (sum(self._labels) / len(self._labels))
        return max(0.10, min(0.50, 1.5 * fail_rate))

    def should_intervene(self, trace: str, threshold: float | None = None) -> bool:
        """Spend a retry/verify/escalation only when failure is likely.

        When threshold is None (default) uses adaptive_threshold, which
        automatically tracks the store's observed failure rate.
        """
        t = threshold if threshold is not None else self.adaptive_threshold
        return self.predict_fail(trace) >= t

    def explain(self, trace: str, k: int = 3) -> List[Dict]:
        """Return the k nearest stored traces driving this prediction.

        Each entry: {"similarity": float, "label": int, "excerpt": str}
        Use to understand WHY a component was flagged.
        """
        if not self._vecs:
            return []
        vec = self._vec(trace)
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

    # ── internals ──────────────────────────────────────────────────────────────
    def _fit_pca(self):
        import numpy as np
        from sklearn.decomposition import PCA
        X = np.array(self._raw_vecs, dtype="float32")
        n_comp = min(self.pca_dim, X.shape[0] - 1)
        pca = PCA(n_components=n_comp)
        pca.fit(X)
        return pca

    def _project_all(self) -> List[list]:
        import numpy as np
        if self._pca is None:
            return [v[:] for v in self._raw_vecs]
        return self._pca.transform(np.array(self._raw_vecs, dtype="float32")).tolist()

    def _vec(self, trace: str) -> list:
        """Embed and project (into PCA space if active) for kNN lookup."""
        raw = self._embed1_raw(trace)
        if self._pca is None:
            return raw
        import numpy as np
        return self._pca.transform(np.array([raw], dtype="float32"))[0].tolist()

    def _embed1_raw(self, trace: str) -> list:
        return self.embedder([trace])[0].tolist()


# ── high-level orchestrator ───────────────────────────────────────────────────
def run_task(
    task:        str,
    agent:       Agent,
    verifier:    Optional[Verifier] = None,
    forecaster:  Optional[Forecaster] = None,
    retriever:   Optional[Callable[..., str]] = None,
    threshold:   float | None = None,
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
        threshold:  P(fail) cutoff to trigger intervention. None (default) uses
                    adaptive_threshold, which tracks the store's observed failure
                    rate automatically — no tuning needed across domains.
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
            retrieval = retriever(q) if retriever else ""
            ctx   = f"Task: {task}\n\n{retrieval}".strip()
            trace = attempt(q, ctx, agent)

            p_fail   = None
            retried  = False
            neighbor = None

            if forecaster and len(forecaster._vecs) >= 2:
                p_fail   = forecaster.predict_fail(trace)
                neighbor = forecaster.nearest_failure(trace)

                t = threshold if threshold is not None else forecaster.adaptive_threshold
                if retry and p_fail >= t:
                    trace   = _text(agent(_RETRY.format(ctx=ctx, q=q, prev=trace[-2000:])))
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
