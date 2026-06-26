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

from .forecast import knn_predict_cross, cosine

Agent = Callable[[str], object]            # prompt -> (text, tokens) | text

# ── prompts for the generic operations (no dataset logic) ────────────────────────
_DECOMPOSE = ("List the sub-questions needed to fully answer this task. "
              "Do NOT solve, execute, run code, or use any tools — output plain text only. "
              "Use the MINIMUM number: one sub-question unless parts require completely "
              "separate knowledge or context. A single program, calculation, or chain of "
              "reasoning — even with multiple outputs — is ONE sub-question. "
              "One sub-question per line, no numbering, no explanations.\n\nTask: {task}")
_ATTEMPT = ("{ctx}\n\nQuestion: {q}\n\n"
            "Reason step by step, noting what you look for and what you find. "
            "If you write code, put the COMPLETE, self-contained solution in ONE "
            "python_exec call (variables do NOT persist between calls) and run it "
            "once to confirm — do not re-implement or re-verify what already works. "
            "End with 'ANSWER: ...'.")
_RETRY  = ("{ctx}\n\nQuestion: {q}\n\nYour previous attempt (which may be wrong):\n{prev}\n\n"
           "First, in one sentence quote the exact step above that is wrong or incomplete "
           "and state what NOT to do. Then reattempt from scratch and end with 'ANSWER: ...'.")
_JUDGE = ("Gold answer to the overall task:\n{gold}\n\nSub-question: {q}\nProposed answer: "
          "{a}\n\nIs the proposed answer correct and supported by the gold? Reply YES or NO.")


def _text(out: object) -> str:
    return str(out[0]) if isinstance(out, tuple) else str(out)


# ── generic task operations ──────────────────────────────────────────────────────
def decompose(task: str, agent: Agent, cap: int = 8) -> List[str]:
    """Split any task into atomic, independently-answerable sub-questions."""
    lines = _text(agent(_DECOMPOSE.format(task=task))).splitlines()
    qs = [
        ln.strip(" -*\t").strip() for ln in lines
        if len(ln.strip()) > 6
        and not ln.strip().startswith("[tool:")   # strip tool-call artifacts
        and "] →" not in ln                       # strip tool result lines
    ]
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
        # Judge the FULL trace, not just the tail. The evidence a judge needs to
        # confirm a claimed "PASS" — the executed code and tool output — usually
        # sits in the middle of a tool-agent trace, while the tail is only closing
        # prose. A small tail window hides that evidence, so the judge rejects the
        # conclusion as unsupported and a correct component is marked FAIL. Cap to
        # the last 16k chars purely as a runaway guard on pathological traces.
        view = answer if len(answer) <= 16000 else answer[-16000:]
        out = _text(judge_agent(_SELF_JUDGE.format(ev=ev, q=question, a=view)))
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


def extract_code(text: str, want_fn: str | None = None) -> str:
    """Extract the best Python code from an agent trace.

    Handles three formats in priority order:
      1. python_exec tool calls  (JSON: ``python_exec({"code": "..."})`` )
      2. Markdown fenced blocks  (``` python ... ```)
      3. Bare def/class blocks   (last resort; requires want_fn)

    Among tool calls the *last* one wins — it is most likely the
    post-brain-corrected version.
    """
    import json as _json, ast as _ast_mod
    candidates: list[tuple[int, str]] = []

    # 1. python_exec JSON tool calls
    call_idx = 0
    for prefix in ('python_exec({"', "python_exec({'"):
        search_pos = 0
        while True:
            start = text.find(prefix, search_pos)
            if start == -1:
                break
            arg_start = start + len("python_exec(")
            depth, i = 0, arg_start
            while i < len(text):
                if text[i] == "{":    depth += 1
                elif text[i] == "}":
                    depth -= 1
                    if depth == 0:
                        break
                i += 1
            raw = text[arg_start: i + 1]
            search_pos = i + 1
            code = None
            for loader in (_json.loads, _ast_mod.literal_eval):
                try:
                    d = loader(raw)
                    if isinstance(d, dict):
                        code = d.get("code") or d.get("script") or d.get("source")
                        if code:
                            break
                except Exception:
                    pass
            if code:
                candidates.append((10 + call_idx, code))
                call_idx += 1

    # 2. Markdown fenced blocks
    for m in re.finditer(r"```(?:python|py)?\s*\n?(.*?)```", text, re.DOTALL | re.I):
        candidates.append((5, m.group(1).strip()))

    # 3. Bare def/class (only when a specific function name is requested)
    if want_fn:
        for kw in ("def", "class"):
            for m in re.finditer(
                rf"({kw} {re.escape(want_fn)}[\s\(].*?)(?=\n(?:class |def )|\Z)",
                text, re.DOTALL,
            ):
                candidates.append((1, m.group(1).strip()))

    if not candidates:
        return text.strip()

    def _score(item: tuple[int, str]) -> tuple[int, int]:
        priority, c = item
        try:
            compile(c, "<s>", "exec")
        except SyntaxError:
            return (-1, 0)
        bonus = 2 if (want_fn and (f"def {want_fn}" in c or f"class {want_fn}" in c)) \
                else (1 if ("def " in c or "class " in c) else 0)
        return (bonus, priority)

    best = max(candidates, key=_score)
    return best[1] if _score(best)[0] >= 0 else text.strip()


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

    The forecaster still wraps the verifier: when the debugging trace resembles
    past failures in the store, P(fail) rises and a retry fires *before* the
    verifier is called, letting the agent self-correct.
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
        """Probability this trace's component FAILS (0..1) — kNN over the store.

        Returns the fraction of the k nearest stored traces that failed: a trace
        landing among past failures scores high, one among past successes scores
        low. That is the whole signal — no heuristics layered on top.

        When the store has no usable signal yet — empty, or only one outcome
        class seen so far — the forecaster ABSTAINS (returns 0.0). You cannot
        forecast failure before you have seen both passes and failures, and
        retrying every component is worse than retrying none. Predictions become
        meaningful once the store holds a mix of both (~50+ traces in practice).
        """
        if len(set(self._labels)) < 2:
            return 0.0
        vec      = self._vec(trace)
        raw_succ = knn_predict_cross(self._vecs, self._labels, [vec], self.k)[0]
        return 1.0 - raw_succ

    @property
    def adaptive_threshold(self) -> float:
        """P(fail) cutoff — 20% of the way from the store's failure rate toward 1.0.

        Below 10 traces: 0.5 (coin-flip baseline, not enough data to calibrate).
        After that: fail_rate + (1 − fail_rate) × 0.20, capped at 0.80.

        This scales correctly at both extremes:
          Low fail (10%):  threshold ≈ 0.28 — catches a 0.35 signal (3× base rate) ✓
          High fail (70%): threshold ≈ 0.76 — only extreme outliers fire, not everything ✓
          High fail (90%): threshold ≈ 0.92 — very selective when almost all fail ✓

        A flat 0.35 triggers on everything in a high-failure store. Setting threshold
        equal to fail_rate means nothing ever triggers (average trace = threshold).
        The "20% toward 1" formula keeps the bar meaningful in both regimes.
        Override per-call with should_intervene(t, threshold=…) to fix the bar.
        """
        n = len(self._labels)
        if n < 10:
            return 0.5
        fail_rate = sum(1 for l in self._labels if l == 0) / n
        return min(fail_rate + (1 - fail_rate) * 0.20, 0.80)

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
    task:             str,
    agent:            Agent,
    verifier:         Optional[Verifier] = None,
    forecaster:       Optional[Forecaster] = None,
    retriever:        Optional[Callable[..., str]] = None,
    threshold:        float | None = None,
    cap:              int = 8,
    display:          bool = True,
    retry:            bool = True,
    retry_agent:      Optional[Agent] = None,
    decompose_agent:  Optional[Agent] = None,
    monitor=None,
    brain=None,
) -> TaskResult:
    """Run the full trace_use pipeline on any task.

    decompose → attempt → [forecast] → [intervene+retry] → [verify] → store

    Args:
        task:            Any natural-language task.
        agent:           Callable (prompt -> text | (text, tokens)). Use haiku,
                         opus, or tool_agent() from agents.py.
        verifier:        Optional (question, answer) -> 0..1 scorer. If omitted,
                         self_judge is used when a forecaster is present.
        forecaster:      Fitted or empty Forecaster. If None, no failure prediction
                         is run — useful for bootstrapping a trace store.
        retriever:       Optional retrieval fn (query -> context string).
        threshold:       P(fail) cutoff to trigger intervention. None (default) uses
                         adaptive_threshold, which tracks the store's observed failure
                         rate automatically — no tuning needed across domains.
        cap:             Max sub-questions to decompose into (default 8).
        display:         Show the live Rich terminal display (default True).
        retry:           Retry once on predicted failure (default True).
        retry_agent:     Agent to use for retries. Defaults to `agent`. Pass a
                         stronger model (e.g. Sonnet) so retries actually improve
                         outcomes rather than repeating the same mistake.
        decompose_agent: Agent to use for task decomposition. Defaults to `agent`.
                         Pass a plain text agent (e.g. haiku) to avoid tool-use
                         during decompose — tool_agent can execute code even when
                         instructed not to, garbling the sub-question list.
        brain:           Optional BrainAgent. When supplied, wires as a monitor AND
                         stores verified node-level patterns after each component.
                         Injects targeted warnings into retry prompts when a failure
                         pattern is detected mid-generation.

    Returns:
        TaskResult with all component traces, labels, predictions, and neighbors.
    """
    from .display import TraceDisplay, print_summary

    store_n    = len(forecaster._vecs) if forecaster else 0
    agent_name = getattr(agent, "__name__", "agent")
    result     = TaskResult(task=task)

    disp = TraceDisplay(task, agent_name=agent_name, store_size=store_n)

    with disp:
        sub_qs = decompose(task, decompose_agent or agent, cap=cap)
        disp.set_components(sub_qs)

        for i, q in enumerate(sub_qs):
            disp.set_attempting(i)
            retrieval = retriever(q) if retriever else ""
            ctx = f"Task: {task}\n\n{retrieval}".strip()

            # Reset the background monitor / brain for each fresh component attempt
            # so the buffer and bail flag from the previous sub-question don't bleed.
            _active_monitor = brain or monitor
            if _active_monitor:
                _active_monitor.reset()
                if hasattr(agent, 'monitor'):
                    agent.monitor = _active_monitor

            trace = attempt(q, ctx, agent)
            first_trace = trace   # preserve first-attempt trajectory for the forecaster

            p_fail   = None
            retried  = False
            neighbor = None

            # Brain bail: if the brain fired mid-generation, seed the retry with the
            # targeted warning so the agent gets a specific diagnosis, not a generic nudge.
            brain_fired = brain and brain.should_bail
            brain_hint  = (brain.get_intervention() or "") if brain_fired else ""

            if forecaster and len(forecaster._vecs) >= 2:
                p_fail   = forecaster.predict_fail(first_trace)
                neighbor = forecaster.nearest_failure(first_trace)

                t = threshold if threshold is not None else forecaster.adaptive_threshold
                should_retry = retry and (p_fail >= t or brain_fired)
                if should_retry:
                    ra = retry_agent or agent
                    if _active_monitor:
                        _active_monitor.reset()
                        if hasattr(ra, 'monitor'):
                            ra.monitor = _active_monitor
                    prev_ctx = (
                        (brain_hint + "\n\n") if brain_hint else ""
                    ) + first_trace[-2000:]
                    trace   = _text(ra(_RETRY.format(ctx=ctx, q=q, prev=prev_ctx)))
                    retried = True
            elif brain_fired and retry:
                # Brain fired but no forecaster — still retry with brain warning
                ra = retry_agent or agent
                if _active_monitor:
                    _active_monitor.reset()
                    if hasattr(ra, 'monitor'):
                        ra.monitor = _active_monitor
                prev_ctx = (
                    (brain_hint + "\n\n") if brain_hint else ""
                ) + first_trace[-2000:]
                trace   = _text(ra(_RETRY.format(ctx=ctx, q=q, prev=prev_ctx)))
                retried = True

            # verify / label — label reflects final outcome (pass/fail after any retry)
            if verifier:
                label = int(verifier(q, trace) >= 0.5)
            else:
                label = 1   # optimistic default when no verifier

            # Store the FIRST ATTEMPT trace with its own label so the forecaster
            # learns which first-attempt trajectory patterns predict failure.
            # Storing the retry trace instead inverts the signal: retry traces
            # have a completely different structure and pollute the kNN store.
            if forecaster:
                first_label = int(verifier(q, first_trace) >= 0.5) if verifier and retried else label
                forecaster.add(first_trace, first_label)
                disp.update_store(len(forecaster._vecs))

            # Brain: finalize parse the completed trace (in case background thread
            # didn't have time), then store node patterns with their verified label.
            if brain:
                first_label_for_brain = (
                    int(verifier(q, first_trace) >= 0.5) if verifier and retried else label
                )
                brain.finalize(first_trace)
                brain.store_result(first_label_for_brain)

            disp.set_result(i, p_fail if p_fail is not None else 0.0,
                            label, retried=retried, neighbor=neighbor)

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
