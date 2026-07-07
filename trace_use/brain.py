"""brain.py — Trajectory-aware failure predictor for any LLM agent.

ARCHITECTURE
────────────
Each agent run produces a trajectory: an ordered sequence of text chunks
(reasoning steps, tool calls, results). Each chunk is embedded with a local
model (sentence-transformers, free, no API key, ~10ms/chunk on CPU).
The brain stores labeled trajectories (pass=1 / fail=0) and uses two signals
to predict failure in real time:

1. PREFIX KNN
   Computes the mean embedding of the live trajectory prefix and the same-
   length prefix of each stored run. kNN over these prefix means → P(fail).
   Works from the very first stored run. No training required.
   Captures: "this reasoning path resembles ones that previously failed."

2. MARKOV STATE FAILURE RATE  (activates once ≥ 30 chunks are stored)
   k-means discretizes all stored chunk embeddings into thought states (k=12).
   Each state tracks what fraction of runs that visited it eventually failed.
   At inference, the current chunk → nearest state → P(fail | state).
   Captures: "models reasoning this way tend to get the wrong answer."

These two signals are combined into a single P(fail). When P(fail) ≥ threshold:
  - For tool-use agents:  brain modifies the tool result to inject a warning
  - For streaming agents: brain fires should_bail, triggering a retry

CODE-SNIPPET STORE (on_tool_call)
──────────────────────────────────
For tool agents, each python_exec call also hits a separate code-snippet kNN
store (FailureStore). This catches bugs deterministically via probe tests and
via kNN over past code snippets — faster than waiting for trajectory signal.

PUBLIC API
──────────
  brain = BrainAgent(embedder)           # build_embedder() from agents.py
  brain.set_task(idx, probe_fn=fn)       # register current task + optional probe
  brain.on_tool_call(name, inp, result)  # intercept tool execution (raw dict)
  brain.store(trace, label, meta)        # store completed run trajectory
  brain.store_code(code, label, meta)    # store code-snippet pass/fail
  brain.store_passing_code(label, code)  # record working example
  brain.get_trajectory()                 # list[TrajectoryPoint] for visualization
"""
from __future__ import annotations

import io
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Callable

import numpy as np


CHUNK_CHARS = 400   # max characters per trajectory chunk


# ── Data types ────────────────────────────────────────────────────────────────

@dataclass
class TrajectoryPoint:
    """One tool-call observation within a task (for the live heatmap)."""
    task_idx:    int
    turn:        int
    p_fail:      float | None
    probe_fails: list[str]
    fired:       bool
    ts:          float = field(default_factory=time.time)

    @property
    def severity(self) -> float:
        probe = 1.0 if self.probe_fails else 0.0
        knn   = self.p_fail if self.p_fail is not None else 0.0
        return max(probe, knn)


@dataclass
class _StoredRun:
    """One completed run stored as a matrix of chunk embeddings."""
    trace_id: str
    label:    int           # 1 = pass, 0 = fail
    vecs:     np.ndarray   # (n_chunks, embed_dim)
    metadata: str = ""


@dataclass
class _StoredCode:
    """One code snippet stored for kNN (used by FailureStore)."""
    trace_id: str
    label:    int
    excerpt:  str
    tail:     str
    vec:      np.ndarray = field(repr=False)
    metadata: str = ""


@dataclass
class _FailurePattern:
    """One extracted logical failure reason with its embedding."""
    reason:  str          # "agent defined Flask routes but never created HTML templates"
    signal:  str          # "what to look for in future runs to detect this recurrence"
    vec:     np.ndarray   # embedding of SIGNAL only — describes code patterns, matches code
    task:    str = ""


# ── Logical failure store ─────────────────────────────────────────────────────

class LogicalFailureStore:
    """Stores LLM-extracted failure REASONS, not trace embeddings.

    When a task fails, an LLM call extracts the specific logical error —
    'agent defined Flask routes but never created the HTML template' —
    and stores its embedding. Future runs are checked against these extracted
    reasons, not against raw trace text.

    This is logic detection, not keyword matching:
    - Trajectory kNN sees: 'these sentences are similar' (surface form)
    - LogicalFailureStore sees: 'this code is missing the same step that
      caused failure before' (causal structure)
    """

    _EXTRACT_PROMPT = (
        "An LLM agent failed the following task. Analyze the trace and identify "
        "the specific logical error, missing step, or wrong assumption that caused "
        "the WRONG ANSWER or WRONG IMPLEMENTATION — not execution problems like "
        "empty tool calls, truncation, or stalling.\n\n"
        "IMPORTANT: Only extract domain-specific logic errors. Examples of what to extract:\n"
        "  - 'Annualization used sqrt(365) instead of sqrt(252) for daily returns'\n"
        "  - 'Tax bracket calculation applied the top rate to the entire income instead of just the top band'\n"
        "  - 'Drawdown peak was not reset after a new high was reached'\n"
        "If the failure is just empty code / truncation / stalling — reply with SKIP.\n\n"
        "Task: {task}\n\n"
        "Trace (last 2000 chars):\n{trace}\n\n"
        "Reply in EXACTLY this format:\n"
        "REASON: <one sentence — the specific algorithmic/logical error>\n"
        "SIGNAL: <one sentence — what to watch for in future code to catch this same error>"
    )

    def __init__(self, embedder: Callable, threshold: float = 0.45):
        self._embedder  = embedder
        self._threshold = threshold
        self._patterns: list[_FailurePattern] = []
        self._lock      = threading.Lock()

    def extract_and_store(
        self, trace: str, task: str = "",
        model: str = "claude-haiku-4-5-20251001",
    ) -> str | None:
        """Call an LLM to extract the failure reason, then store it.

        Runs in a background thread so it never blocks the session loop.
        Returns the extracted reason string, or None on failure.
        """
        from . import agents as _ag
        prompt = self._EXTRACT_PROMPT.format(
            task=task[:300],
            trace=trace[-2000:],
        )
        try:
            raw, _ = _ag._anthropic_call(model, prompt, max_tokens=200)
            if raw.strip().upper().startswith("SKIP"):
                return None   # stall/execution failure — not a logic pattern worth storing
            reason = signal = ""
            for line in raw.splitlines():
                if line.startswith("REASON:"):
                    reason = line[len("REASON:"):].strip()
                elif line.startswith("SIGNAL:"):
                    signal = line[len("SIGNAL:"):].strip()
            if not reason or not signal:
                return None
            # Embed the SIGNAL only — it describes code patterns to watch for,
            # so it lives in the same semantic space as future agent code/output.
            vec = np.asarray(self._embedder([signal])[0], dtype="float32")
            with self._lock:
                self._patterns.append(_FailurePattern(
                    reason=reason, signal=signal, vec=vec, task=task[:120],
                ))
            return reason
        except Exception:
            return None

    def query(self, text: str) -> tuple[str | None, float | None]:
        """Check whether text matches a known logical failure pattern.

        Returns (warning_message, similarity) or (None, None).
        The warning names the specific error, not just 'reconsider approach'.
        """
        with self._lock:
            if not self._patterns:
                return None, None
            patterns = list(self._patterns)

        vec  = np.asarray(self._embedder([text[-1500:]])[0], dtype="float32")
        best_sim, best = max(
            ((float(np.dot(vec, p.vec)), p) for p in patterns),
            key=lambda x: x[0],
        )
        if best_sim < self._threshold:
            return None, None

        msg = (
            f"Known failure pattern detected (similarity {best_sim:.0%}):\n"
            f"  WHAT WENT WRONG BEFORE: {best.reason}\n"
            f"  WATCH FOR: {best.signal}"
        )
        return msg, best_sim

    @property
    def n_patterns(self) -> int:
        with self._lock:
            return len(self._patterns)


# ── Trajectory store ──────────────────────────────────────────────────────────

class TrajectoryStore:
    """Stores labeled agent runs as ordered chunk-embedding sequences.

    Provides two P(fail) signals:
      1. Prefix kNN       — similarity of the live trajectory prefix to stored ones
      2. Markov state FR  — failure rate of the nearest thought-state cluster
    """

    def __init__(self, embedder: Callable, k: int = 5,
                 threshold: float = 0.45, n_clusters: int = 12):
        self._embedder   = embedder
        self._k          = k
        self._threshold  = threshold
        self._n_clusters = n_clusters
        self._runs:      list[_StoredRun] = []
        self._lock       = threading.Lock()

        # Markov chain (fitted lazily once enough data exists)
        self._km                       = None   # sklearn MiniBatchKMeans or None
        self._fail_rate: np.ndarray | None = None  # shape (n_clusters,)
        self._all_vecs:  list[np.ndarray]  = []    # every chunk vec from every run
        self._all_run_ids: list[int]       = []    # which run each vec belongs to
        self._markov_dirty: bool           = False

    # ── storage ───────────────────────────────────────────────────────────────

    def add(self, trace: str, label: int, metadata: str = "") -> None:
        """Chunk a completed trace, embed all chunks, store with label."""
        chunks = _chunk_trace(trace)
        if not chunks:
            return
        vecs = self._embedder([c[:1500] for c in chunks])
        run_idx = len(self._runs)
        entry = _StoredRun(
            trace_id=uuid.uuid4().hex[:8],
            label=label,
            vecs=np.asarray(vecs, dtype="float32"),
            metadata=metadata,
        )
        with self._lock:
            self._runs.append(entry)
            for v in entry.vecs:
                self._all_vecs.append(v)
                self._all_run_ids.append(run_idx)
            self._markov_dirty = True

        # Lazily refit Markov chain when threshold crossed
        n_chunks = len(self._all_vecs)
        n_runs   = len(self._runs)
        if n_chunks >= self._n_clusters * 3 and n_runs >= 5 and self._markov_dirty:
            self._refit_markov()

    # ── prediction ────────────────────────────────────────────────────────────

    def predict(self, partial_trace: str) -> tuple[float | None, str | None]:
        """Predict P(fail) for a partial trace. Returns (p_fail, warning_or_None)."""
        with self._lock:
            runs      = list(self._runs)
            km        = self._km
            fail_rate = self._fail_rate

        if not runs or len(set(r.label for r in runs)) < 2:
            return None, None

        chunks = _chunk_trace(partial_trace)
        if not chunks:
            return None, None
        live_vecs = np.asarray(
            self._embedder([c[:1500] for c in chunks]), dtype="float32"
        )
        if live_vecs.ndim < 2 or len(live_vecs) == 0:
            return None, None

        # Signal 1: prefix kNN
        p_prefix, best_fail = self._prefix_knn(live_vecs, runs)

        # Signal 2: Markov state failure rate
        p_markov: float | None = None
        if km is not None and fail_rate is not None:
            state    = int(km.predict(live_vecs[-1:, :])[0])
            p_markov = float(fail_rate[state])

        # Combine signals
        if p_prefix is not None and p_markov is not None:
            p_fail = 0.55 * p_markov + 0.45 * p_prefix
        elif p_markov is not None:
            p_fail = p_markov
        else:
            p_fail = p_prefix

        if p_fail is None or p_fail < self._threshold:
            return p_fail, None

        return p_fail, _build_traj_warning(p_fail, best_fail)

    def _prefix_knn(
        self, live_vecs: np.ndarray, runs: list[_StoredRun]
    ) -> tuple[float, _StoredRun | None]:
        """kNN over prefix-mean embeddings. Returns (p_fail, best_failing_run)."""
        L = len(live_vecs)
        live_mean = live_vecs.mean(axis=0)
        live_mean = live_mean / (np.linalg.norm(live_mean) + 1e-9)

        prefix_means = []
        for run in runs:
            n = min(L, len(run.vecs))
            m = run.vecs[:n].mean(axis=0)
            m = m / (np.linalg.norm(m) + 1e-9)
            prefix_means.append(m)

        mat  = np.stack(prefix_means)           # (n_runs, dim)
        sims = mat @ live_mean                  # cosine similarity
        k    = min(self._k, len(runs))
        top  = np.argpartition(sims, -k)[-k:]

        # Only use neighbours that are actually similar (min cosine ≥ 0.55).
        # Without this floor, every query matches the top-k regardless of distance,
        # causing false positives once enough runs are stored.
        top = [i for i in top if sims[i] >= 0.55]
        if not top:
            return 0.0, None

        n_fail = sum(1 for i in top if runs[i].label == 0)
        p_fail = n_fail / len(top)

        best: _StoredRun | None = None
        fail_pairs = [(runs[i], float(sims[i])) for i in top if runs[i].label == 0]
        if fail_pairs:
            best = max(fail_pairs, key=lambda x: x[1])[0]

        return p_fail, best

    # ── Markov chain fitting ──────────────────────────────────────────────────

    def _refit_markov(self) -> None:
        """Fit k-means over all stored chunk embeddings; compute per-cluster failure rates."""
        try:
            from sklearn.cluster import MiniBatchKMeans
        except ImportError:
            return  # scikit-learn not available — prefix kNN only

        with self._lock:
            all_vecs  = np.stack(self._all_vecs)
            run_ids   = list(self._all_run_ids)
            runs      = list(self._runs)
            self._markov_dirty = False

        k = min(self._n_clusters, max(2, len(all_vecs) // 3))
        km = MiniBatchKMeans(n_clusters=k, random_state=42, n_init=3)
        states = km.fit_predict(all_vecs)

        # Failure rate per cluster: P(fail | visited this state)
        fail_count  = np.zeros(k)
        visit_count = np.zeros(k)
        for chunk_idx, cluster_id in enumerate(states):
            run_label = runs[run_ids[chunk_idx]].label
            fail_count[cluster_id]  += 1 - run_label   # fail=1, pass=0
            visit_count[cluster_id] += 1

        with np.errstate(divide='ignore', invalid='ignore'):
            fail_rate = np.where(visit_count > 0, fail_count / visit_count, 0.5)

        with self._lock:
            self._km        = km
            self._fail_rate = fail_rate

    # ── stats ─────────────────────────────────────────────────────────────────

    @property
    def n(self)      -> int: return len(self._runs)
    @property
    def n_fail(self) -> int: return sum(1 for r in self._runs if r.label == 0)
    @property
    def n_pass(self) -> int: return sum(1 for r in self._runs if r.label == 1)

    def all_vecs(self) -> tuple[np.ndarray | None, list[int]]:
        """First-chunk embedding per run + labels (for visualization)."""
        with self._lock:
            if not self._runs:
                return None, []
            mat    = np.stack([r.vecs[0] for r in self._runs])
            labels = [r.label for r in self._runs]
        return mat, labels


# ── Code-snippet kNN store ────────────────────────────────────────────────────

class FailureStore:
    """kNN store over code snippets. Used by on_tool_call for fast per-turn checks."""

    def __init__(self, embedder: Callable, k: int = 5, threshold: float = 0.45):
        self._embedder  = embedder
        self._k         = k
        self._threshold = threshold
        self._items:    list[_StoredCode] = []
        self._matrix:   np.ndarray | None = None
        self._lock      = threading.Lock()

    def add(self, code: str, label: int, metadata: str = "") -> None:
        if not code.strip():
            return
        vec = self._embedder([code[-1500:]])[0]
        entry = _StoredCode(
            trace_id=uuid.uuid4().hex[:8],
            label=label,
            excerpt=code[:300],
            tail=code[-300:],
            vec=np.asarray(vec, dtype="float32"),
            metadata=metadata,
        )
        with self._lock:
            self._items.append(entry)
            self._matrix = np.stack([i.vec for i in self._items])

    def query(self, code: str) -> tuple[float | None, str | None]:
        with self._lock:
            if self._matrix is None:
                return None, None
            labels = [i.label for i in self._items]
            if len(set(labels)) < 2:
                return None, None
            mat   = self._matrix.copy()
            items = list(self._items)

        vec  = self._embedder([code[-1500:]])[0]
        sims = mat @ vec
        k    = min(self._k, len(items))
        top  = np.argpartition(sims, -k)[-k:]

        n_fail = sum(1 for i in top if items[i].label == 0)
        p_fail = n_fail / k

        if p_fail < self._threshold:
            return p_fail, None

        fail_nbrs = sorted(
            [(items[i], float(sims[i])) for i in top if items[i].label == 0],
            key=lambda x: x[1], reverse=True,
        )
        if not fail_nbrs:
            return p_fail, None

        lines = [f"[BRAIN — P(fail)={p_fail:.0%}]  Similar code previously failed:"]
        seen: set[str] = set()
        for item, _ in fail_nbrs[:2]:
            if item.metadata and item.metadata not in seen:
                seen.add(item.metadata)
                lines.append(f"  • {item.metadata[:160]}")
        if not seen:
            lines.append(f"  Excerpt: \"{fail_nbrs[0][0].excerpt[:160]}\"")
        lines.append("Avoid the same mistake.")
        return p_fail, "\n".join(lines)

    @property
    def n(self) -> int: return len(self._items)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _chunk_trace(trace: str) -> list[str]:
    """Split a trace into ordered semantic chunks.

    Chunks at tool call boundaries first, then paragraph breaks within each
    part, then hard-caps at CHUNK_CHARS to keep embeddings well-conditioned.
    """
    raw_parts = re.split(r'(?=\[tool:)', trace)
    chunks: list[str] = []
    for part in raw_parts:
        for para in part.split('\n\n'):
            para = para.strip()
            if not para:
                continue
            if len(para) <= CHUNK_CHARS:
                chunks.append(para)
            else:
                for start in range(0, len(para), CHUNK_CHARS):
                    chunk = para[start:start + CHUNK_CHARS].strip()
                    if chunk:
                        chunks.append(chunk)
    return chunks


def _build_traj_warning(
    p_fail: float,
    best_run: "_StoredRun | None",
    logic_reason: str | None = None,
) -> str:
    lines = [f"[BRAIN — P(fail)={p_fail:.0%}]  Trajectory resembles past failures."]
    if best_run and best_run.metadata:
        lines.append(f"  Most similar failed run: {best_run.metadata[:200]}")
    if logic_reason:
        lines.append(f"  Known failure pattern: {logic_reason}")
    lines.append("Reconsider your current approach before continuing.")
    return "\n".join(lines)


def _inject_cot(prompt: str) -> str:
    _code_kw = ("function", "class", "implement", "write a", "def ", "code")
    is_code = any(kw in prompt.lower() for kw in _code_kw)
    suffix = (
        "\n\nWork through your approach step by step, explaining your reasoning. "
        "Then provide the complete implementation in a ```python code block."
        if is_code else
        "\n\nWork through this step by step, showing each reasoning step. "
        "Give your final answer as 'ANSWER: ...'."
    )
    return prompt + suffix


# ── BrainAgent ────────────────────────────────────────────────────────────────

class BrainAgent:
    """Inference-time failure detector. Wraps any agent or model.

    Usage::
        from agents import build_embedder, tool_agent
        from brain import BrainAgent

        brain = BrainAgent(build_embedder())

        # Tool agent (probe tests + trajectory + kNN)
        agent = tool_agent(["python_exec"])
        agent.monitor = brain
        brain.set_task(idx, probe_fn=my_probe)
        brain.reset()
        result = agent(prompt)
        brain.store(trace, label, metadata)

        # Any other agent
        agent = brain.wrap_any("claude-haiku-4-5-20251001")
        result = agent(prompt)
        brain.store(result[0], label)
    """

    _RETRY_PREFIX = (
        "STOP. The brain detected your approach matches past failures.\n\n"
        "{warning}\n\n"
        "Apply the specific fixes above. "
        "Write the correct, complete implementation in a single ```python code block.\n\n"
        "Task:\n"
    )

    def __init__(self, embedder: Callable, k: int = 5, threshold: float = 0.45,
                 check_interval: float = 1.0, min_chars: int = 200):
        self._embedder       = embedder
        self._check_interval = check_interval
        self._min_chars      = min_chars

        # Three stores: trajectory-level, code-snippet-level, logical failure reasons
        self._traj_store    = TrajectoryStore(embedder, k=k, threshold=threshold)
        self._code_store    = FailureStore(embedder, k=k, threshold=threshold)
        self._logic_store   = LogicalFailureStore(embedder)   # LLM-extracted failure reasons
        self._current_task_str: str = ""   # task description for failure extraction

        # Streaming buffer (push/should_bail interface for tool_agent / streaming_agent)
        self._buffer  = ""
        self._lock    = threading.Lock()
        self._bail_ev = threading.Event()
        self._stop_ev = threading.Event()
        self._thread: threading.Thread | None = None

        # State for current run
        self.last_p_fail:         float | None = None
        self.last_warning:        str   | None = None
        self._pending_trace:      str          = ""
        self._last_code_warning:  str          = ""
        self._code_interventions: int          = 0

        # Turn tracking — lets callers see when the brain fired vs when stall started
        self._turn_count:         int          = 0   # tool calls so far this task
        self._fire_turn:          int | None   = None  # turn on which brain first fired

        # Stall detector — fires when agent makes consecutive empty/no-output tool calls
        self._stall_streak:       int          = 0   # consecutive unproductive calls
        self._stall_threshold:    int          = 2   # fire after this many in a row
        self._stall_fired:        bool         = False

        # Trajectory visualization
        self._trajectory:   list[TrajectoryPoint] = []
        self._traj_lock     = threading.Lock()
        self._task_probe_fn: Callable | None      = None
        self._current_task:  int                  = 0
        self._current_turn:  int                  = 0
        self._passing_codes: list[tuple[str, str]] = []

    # ── task registration ─────────────────────────────────────────────────────

    def set_task(self, task_idx: int, probe_fn: Callable | None = None,
                 task: str = "") -> None:
        """Register the current task index and optional deterministic probe.

        probe_fn(ns: dict) -> list[str]
          Receives the exec'd namespace after python_exec. Returns a list of
          failure strings (empty = all probes passed).

        Call this before each task.
        """
        self._current_task       = task_idx
        self._current_turn       = 0
        self._task_probe_fn      = probe_fn
        self._code_interventions = 0

    # ── on_tool_call: core intervention hook ──────────────────────────────────

    def on_tool_call(self, name: str, input_dict: dict, result: str) -> str | None:
        """Intercept every tool execution — any tool, any task type.

        Returns a modified result string when the brain intervenes, or None to
        leave the result unchanged. Fires at most 2 times per task.

        Signals (in priority order):
          1. Stall detector    — consecutive empty/no-output calls (any tool, no prior data needed)
          2. Probe tests       — deterministic unit tests (python_exec only)
          3. Logical failure   — LLM-extracted reasons from past failures (any tool, any task)
          4. Code snippet kNN  — past code similarity (python_exec only)
          5. Trajectory kNN    — partial trace vs past run prefixes (any tool, any task type)
        """
        if self._code_interventions >= 2:
            return None

        self._turn_count += 1
        is_no_output = not result or result.strip() in ("", "(no output)", "None")

        # ── python_exec-specific fields ───────────────────────────────────────
        code = ""
        if name == "python_exec":
            code = (input_dict or {}).get("code", "") if isinstance(input_dict, dict) else ""

        # ── Stall detector (any tool) ─────────────────────────────────────────
        # A call is unproductive if it has no meaningful input or no output.
        no_input = not code if name == "python_exec" else not any(
            str(v).strip() for v in (input_dict or {}).values()
        )
        if no_input or is_no_output:
            self._stall_streak += 1
            if self._stall_streak >= self._stall_threshold and not self._stall_fired:
                self._stall_fired        = True
                self._code_interventions += 1
                if self._fire_turn is None:
                    self._fire_turn = self._turn_count
                msg = (
                    f"[BRAIN — STALL DETECTED after {self._stall_streak} unproductive calls]\n"
                    f"Your last {self._stall_streak} '{name}' calls produced no meaningful "
                    f"output. You are stuck. Stop and reconsider your approach:\n"
                    f"  1. Do not repeat the same call with the same or empty inputs.\n"
                    f"  2. Try a fundamentally different approach or break the problem down.\n"
                    f"  3. If writing code: produce one complete, self-contained implementation.\n"
                    f"Make your next call count."
                )
                self._last_code_warning = msg
                self.last_warning       = msg
                return f"{result}\n\n{msg}" if result else msg
            return None
        else:
            self._stall_streak = 0

        # ── Accumulate partial trace (any tool) ───────────────────────────────
        input_summary = code if name == "python_exec" else str(input_dict or "")[:500]
        self._pending_trace += f"\n[tool:{name}]\n{input_summary}\n[result]\n{result[:1000]}\n"

        # ── Probe + code kNN (python_exec only) ───────────────────────────────
        probe_fails: list[str] = []
        p_code:      float | None = None
        knn_warning: str   | None = None
        if name == "python_exec" and code:
            probe_fails         = self._run_probe(code)
            p_code, knn_warning = self._code_store.query(code[:2000])

        # ── Logical failure pattern check ─────────────────────────────────────
        # Query with the agent's actual code/output — the SIGNAL embeddings describe
        # code patterns ("look for period_return = (end - cashflow) / start"), so they
        # match code text structurally, not topics. This works for any task type.
        if not probe_fails:
            check_text = (code or result[:1500] or "").strip()
            logic_warning, logic_sim = self._logic_store.query(check_text) if check_text else (None, None)
            if logic_warning and not knn_warning:
                knn_warning = logic_warning
                # Convert similarity to a P(fail) estimate
                if p_code is None:
                    p_code = logic_sim

        # ── Trajectory prefix kNN + Markov (any tool, any task type) ─────────
        # Suppress until enough examples exist — below 10 the signal is noise.
        p_traj, traj_warning = (
            self._traj_store.predict(self._pending_trace)
            if len(self._traj_store._runs) >= 10
            else (None, None)
        )

        # Enrich trajectory warning with the closest extracted logic pattern.
        # The logic store holds causal reasons ("TWR formula wrong: (end-cashflow)/start");
        # query it against the current code/result so the PREDICT warning is specific.
        if traj_warning and self._logic_store.n_patterns > 0:
            check = (code or result[:1500] or "").strip()
            if check:
                logic_reason_msg, _ = self._logic_store.query(check)
                if logic_reason_msg:
                    # Append the specific known failure to the trajectory warning
                    traj_warning = traj_warning + f"\n  {logic_reason_msg}"

        # Combine signals
        if p_traj is not None and p_code is not None:
            p_fail      = 0.55 * p_traj + 0.45 * p_code
            knn_warning = traj_warning or knn_warning
        elif p_traj is not None:
            p_fail      = p_traj
            knn_warning = traj_warning
        else:
            p_fail = p_code

        # ── Fire decision ─────────────────────────────────────────────────────
        # Probe is authoritative when registered — empty probe_fails = verified correct.
        # Fall back to kNN when no probe exists.
        if self._task_probe_fn is not None and name == "python_exec":
            fired = bool(probe_fails)
        else:
            fired = p_fail is not None and p_fail >= self._traj_store._threshold

        pt = TrajectoryPoint(
            task_idx=self._current_task,
            turn=self._current_turn,
            p_fail=p_fail,
            probe_fails=probe_fails,
            fired=fired,
        )
        with self._traj_lock:
            self._trajectory.append(pt)
        self._current_turn += 1

        if not fired:
            return None

        if self._fire_turn is None:
            self._fire_turn = self._turn_count   # record which turn the brain first fired

        warning = self._build_code_intervention(probe_fails, p_fail, knn_warning)
        self._last_code_warning  = warning
        self.last_warning        = warning
        self.last_p_fail         = p_fail
        self._code_interventions += 1

        pf_str = f"{p_fail:.0%}" if p_fail is not None else "?"
        suffix = (
            "Fix the specific issue above then call python_exec again with the corrected code."
            if name == "python_exec"
            else "Reconsider your approach based on the warning above before continuing."
        )
        return f"{result}\n\n[BRAIN — P(fail)={pf_str}]\n{warning}\n{suffix}"

    def on_chunk(self, text: str) -> str | None:
        """Hook for text-only tasks with no tool calls.

        Call this with each new chunk of the agent's output (or the full
        response so far). The brain accumulates it and queries the trajectory
        store every 200 tokens to check whether the reasoning looks like a
        past failure. Returns a warning string if P(fail) >= threshold, else None.

        For streaming agents, call on every chunk. For non-streaming, call once
        with the full response so far at natural checkpoints.
        """
        if self._code_interventions >= 2:
            return None

        self._pending_trace += text
        self._current_turn  += 1

        # Only query trajectory store every ~200 tokens to avoid embedding overhead
        if self._current_turn % 5 != 0:
            return None

        p_traj, traj_warning = self._traj_store.predict(self._pending_trace)
        if p_traj is None or p_traj < self._traj_store._threshold:
            return None

        self._last_code_warning  = traj_warning or ""
        self.last_p_fail         = p_traj
        self._code_interventions += 1
        pf_str = f"{p_traj:.0%}"
        return (
            f"\n[BRAIN — P(fail)={pf_str}  trajectory looks like past failures]\n"
            f"{traj_warning or 'This reasoning pattern has preceded failures before.'}\n"
            f"Reconsider your approach."
        )

    def _run_probe(self, code: str) -> list[str]:
        """Execute code and run the registered probe. Returns failure strings."""
        if not self._task_probe_fn:
            return []
        try:
            ns: dict = {}
            with _suppress_stdout():
                exec(compile(code, "<brain_probe>", "exec"), ns)
            result = self._task_probe_fn(ns)
            if isinstance(result, list):
                return result
            return [] if result else ["probe returned False"]
        except Exception as exc:
            return [f"exec error: {exc}"]

    def _build_code_intervention(
        self, probe_fails: list[str], p_fail: float | None, knn_warning: str | None
    ) -> str:
        parts: list[str] = []
        if probe_fails:
            parts.append("STOP — your code fails these tests RIGHT NOW:")
            for f in probe_fails[:2]:
                condensed  = f.split(". FIX:")[0][:120]
                fix_clause = ("  FIX: " + f.split("FIX:", 1)[1].strip()[:120]) if "FIX:" in f else ""
                parts.append(f"  ✗ {condensed}")
                if fix_clause:
                    parts.append(fix_clause)
            parts.append("Fix this issue and call python_exec again with the corrected complete code.")
        elif knn_warning:
            for line in knn_warning.splitlines():
                line = line.strip()
                if line and not line.startswith("[BRAIN") and not line.startswith("Avoid"):
                    parts.append(f"Warning (similar past failure): {line[:160]}")
                    break
        return "\n".join(parts) if parts else "This pattern has caused failures before — double-check correctness."

    # ── wrap: one-line integration for any agent callable ─────────────────────

    def wrap(self, agent_fn: Callable, verifier: Callable | None = None) -> Callable:
        """Wrap any agent callable so the brain monitors every call automatically.

        Works with any callable that takes a prompt string and returns either a
        string or a (string, tokens) tuple — haiku, opus, tool_agent, or your
        own function.

        The wrapped agent:
          - Resets brain state before each call
          - Passes the full response through the trajectory store after each call
          - Stores the result automatically when a verifier is provided
          - Returns the same value as the original agent (transparent wrapper)

        Without a verifier the brain accumulates traces but cannot label them —
        call brain.store(trace, label) manually after evaluating the result.

        Example::

            brain  = BrainAgent(build_embedder())
            agent  = brain.wrap(haiku)                  # text agent
            agent  = brain.wrap(tool_agent(["python_exec"]))  # tool agent

            # with auto-labeling
            judge  = self_judge(judge_agent=opus)
            agent  = brain.wrap(haiku, verifier=judge)
        """
        task_counter = [0]

        def wrapped(prompt: str, **kwargs):
            idx = task_counter[0]
            self.set_task(idx, task=prompt[:300])
            self._current_task_str = prompt[:300]
            self.reset()

            # For tool_agent the monitor hook fires on every tool call automatically.
            # For plain text agents there are no tool calls — attach monitor here so
            # on_tool_call is still wired if the agent supports it.
            if hasattr(agent_fn, "monitor"):
                agent_fn.monitor = self

            result  = agent_fn(prompt, **kwargs)
            trace   = result[0] if isinstance(result, tuple) else result
            tokens  = result[1] if isinstance(result, tuple) else 0

            # Pass the full response through trajectory store for between-task learning.
            # (For tool_agent, on_tool_call already accumulated _pending_trace mid-run;
            #  this call adds any remaining text and ensures the store is up to date.)
            self.on_chunk(trace)

            if verifier is not None:
                label = 1 if verifier(prompt, trace) >= 0.5 else 0
                self.store(trace, label)

            task_counter[0] += 1
            return result

        wrapped._brain      = self       # expose brain for inspection
        wrapped._task_count = task_counter
        return wrapped

    # ── storage ───────────────────────────────────────────────────────────────

    def store(self, trace: str, label: int, metadata: str = "") -> None:
        """Store a completed run's full trace as a labeled trajectory.

        Call after every task. The trace is chunked and embedded; both the
        trajectory store (prefix-kNN + Markov) and the buffer are updated.

        When label=0 (failure), automatically extracts the logical failure reason
        in a background thread so future runs can be warned about the same mistake.
        """
        self._traj_store.add(trace, label, metadata)
        self._pending_trace = trace

        if label == 0:
            task = self._current_task_str
            def _extract():
                self._logic_store.extract_and_store(trace, task=task)
            threading.Thread(target=_extract, daemon=True, name="brain-extract").start()

    def store_code(self, code: str, label: int, metadata: str = "") -> None:
        """Store a code snippet with its pass/fail label.

        Call after each task with the first-attempt code. The brain learns
        which code patterns fail and warns mid-turn on future tasks.
        """
        if code.strip():
            self._code_store.add(code[:2000], label, metadata)

    def store_passing_code(self, label: str, code: str) -> None:
        """Record a verified-passing code snippet as a working example."""
        self._passing_codes.append((label, code))
        if code.strip():
            self._code_store.add(code[:2000], 1, metadata=f"passing: {label}")

    def store_result(self, label: int) -> None:
        """Store the pending trace with a label (BackgroundMonitor-compatible)."""
        if self._pending_trace:
            self.store(self._pending_trace, label)

    def finalize(self, trace: str) -> None:
        self._pending_trace = trace

    # ── trajectory visualization ──────────────────────────────────────────────

    def get_trajectory(self) -> list[TrajectoryPoint]:
        with self._traj_lock:
            return list(self._trajectory)

    # ── BackgroundMonitor-compatible streaming interface ──────────────────────

    def push(self, text: str) -> None:
        with self._lock:
            self._buffer += text

    @property
    def should_bail(self) -> bool:
        return self._bail_ev.is_set()

    def reset(self) -> None:
        with self._lock:
            self._buffer = ""
        self._bail_ev.clear()
        self.last_p_fail         = None
        self.last_warning        = None
        self._pending_trace      = ""
        self._last_code_warning  = ""
        self._code_interventions = 0
        self._current_turn       = 0
        self._turn_count         = 0
        self._fire_turn          = None
        self._stall_streak       = 0
        self._stall_fired        = False

    def get_intervention(self) -> str | None:
        return self.last_warning

    def get_code_warning(self) -> str:
        return self._last_code_warning

    def check_code(self, code: str) -> bool:
        p_fail, warning = self._code_store.query(code[:2000])
        if p_fail is not None:
            self.last_p_fail = p_fail
        if warning:
            self._last_code_warning = warning
            return True
        return False

    def pulse(self) -> bool:
        """Immediate trajectory check against the current buffer.

        Call after a tool event for faster-than-timed-loop detection.
        Returns True if the brain decided to bail.

        Bailing is suppressed until at least 10 trajectories are stored —
        below that threshold the kNN has too few examples to be trustworthy
        and will generate false positives that kill tasks prematurely.
        """
        with self._lock:
            buf = self._buffer
        if len(buf) < self._min_chars:
            return False
        # Don't bail on noise — require enough stored trajectories for signal
        if len(self._traj_store._runs) < 10:
            return False
        p_fail, warning = self._traj_store.predict(buf)
        if p_fail is not None:
            self.last_p_fail = p_fail
        if warning:
            self.last_warning = warning
            self._bail_ev.set()
            return True
        return False

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> "BrainAgent":
        self._stop_ev.clear()
        self._bail_ev.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="brain")
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop_ev.set()
        if self._thread:
            self._thread.join(timeout=3)

    def __enter__(self) -> "BrainAgent":
        return self.start()

    def __exit__(self, *_) -> None:
        self.stop()

    def _loop(self) -> None:
        """Background thread: periodically checks the streaming buffer."""
        while not self._stop_ev.is_set():
            time.sleep(self._check_interval)
            with self._lock:
                buf = self._buffer
            if len(buf) < self._min_chars:
                continue
            p_fail, warning = self._traj_store.predict(buf)
            if p_fail is not None:
                self.last_p_fail = p_fail
            if warning:
                self.last_warning = warning
                self._bail_ev.set()
                return

    # ── wrap_any API ──────────────────────────────────────────────────────────

    def wrap_any(self, agent_or_model, retry_fn=None, cot: bool = False):
        """Wrap any model string or callable with brain monitoring.

        Model routing:
          'claude-sonnet-4-6' / 'claude-opus-4-8' → extended thinking (native CoT)
          Any other string                          → streaming agent
          Any callable                              → callable wrapper
        """
        _THINKING = {"claude-sonnet-4-6", "claude-opus-4-8", "claude-opus-4-7"}
        if isinstance(agent_or_model, str):
            model = agent_or_model
            if any(m in model for m in _THINKING):
                return self._wrap_thinking(model, retry_fn)
            return self._wrap_model(model, retry_fn, cot=cot)
        return self._wrap_callable(agent_or_model, retry_fn, cot=cot)

    def wrap_thinking(self, model: str = "claude-sonnet-4-6",
                      budget: int = 5000, retry_fn=None):
        return self._wrap_thinking(model, retry_fn)

    # ── seed (pre-populate with expert knowledge) ─────────────────────────────

    def seed(self, items: list[dict]) -> None:
        """Pre-populate stores with known failure/pass patterns.

        Each item: {'trace': str, 'label': int, 'metadata': str}
        """
        for item in items:
            self.store(item["trace"], item["label"], item.get("metadata", ""))

    # ── accessors ─────────────────────────────────────────────────────────────

    @property
    def n_stored(self) -> int:  return self._traj_store.n
    @property
    def n_fail(self)   -> int:  return self._traj_store.n_fail
    @property
    def n_pass(self)   -> int:  return self._traj_store.n_pass
    @property
    def failure_store(self):    return self._traj_store   # backward compat

    # ── internal wrap implementations ─────────────────────────────────────────

    def _wrap_callable(self, agent_fn, retry_fn=None, cot: bool = False):
        _retry = retry_fn or agent_fn
        brain  = self

        def monitored(prompt: str):
            brain.reset()
            gen_prompt = _inject_cot(prompt) if cot else prompt
            if hasattr(agent_fn, 'monitor'):
                agent_fn.monitor = brain
            with brain:
                result = agent_fn(gen_prompt)
            if hasattr(agent_fn, 'monitor'):
                agent_fn.monitor = None
            trace = result[0] if isinstance(result, tuple) else str(result)
            brain._pending_trace = trace
            if brain.should_bail:
                warn = brain.last_warning or ""
                rp   = brain._RETRY_PREFIX.format(warning=warn) + gen_prompt if warn else gen_prompt
                return _retry(rp)
            return result

        monitored.__name__ = getattr(agent_fn, '__name__', 'monitored')
        monitored._brain   = brain
        return monitored

    def _wrap_model(self, model: str, retry_fn=None, cot: bool = False):
        from .agents import streaming_agent
        stream = streaming_agent(model)
        brain  = self
        _retry = retry_fn

        def monitored(prompt: str):
            brain.reset()
            gen_prompt = _inject_cot(prompt) if cot else prompt
            stream.monitor = brain
            with brain:
                result = stream(gen_prompt)
            stream.monitor = None
            trace = result[0] if isinstance(result, tuple) else str(result)
            brain._pending_trace = trace
            if brain.should_bail:
                warn = brain.last_warning or ""
                rp   = brain._RETRY_PREFIX.format(warning=warn) + gen_prompt if warn else gen_prompt
                r_fn = _retry or (lambda p: stream(p))
                return r_fn(rp)
            return result

        monitored.__name__ = f"brain({model.split('-')[1] if '-' in model else model})"
        monitored._brain   = brain
        return monitored

    def _wrap_thinking(self, model: str, retry_fn=None):
        """Extended thinking — model's internal CoT streams directly to the brain."""
        import anthropic
        from . import agents as _ag
        brain  = self
        _retry = retry_fn

        def monitored(prompt: str) -> tuple:
            brain.reset()
            if _ag._client is None:
                _ag._client = anthropic.Anthropic()
            client = _ag._client

            thinking_acc = text_acc = ""
            total_tokens = 0
            bailed       = False

            brain.start()
            try:
                with client.messages.stream(
                    model=model,
                    max_tokens=16000,
                    thinking={"type": "enabled", "budget_tokens": 5000},
                    messages=[{"role": "user", "content": prompt}],
                ) as stream:
                    for event in stream:
                        etype = getattr(event, 'type', '')
                        if etype == 'content_block_delta':
                            delta = getattr(event, 'delta', None)
                            if delta is None:
                                continue
                            dtype = getattr(delta, 'type', '')
                            if dtype == 'thinking_delta':
                                chunk = getattr(delta, 'thinking', '')
                                thinking_acc += chunk
                                with brain._lock:
                                    brain._buffer += chunk
                            elif dtype == 'text_delta':
                                chunk = getattr(delta, 'text', '')
                                text_acc += chunk
                                with brain._lock:
                                    brain._buffer += chunk
                        if brain.should_bail:
                            bailed = True
                            break
                if not bailed:
                    try:
                        msg = stream.get_final_message()
                        total_tokens = msg.usage.input_tokens + msg.usage.output_tokens
                    except Exception:
                        pass
            finally:
                brain.stop()

            full_trace = f"[THINKING]\n{thinking_acc}\n[RESPONSE]\n{text_acc}"
            brain._pending_trace = full_trace
            if bailed:
                warn  = brain.last_warning or ""
                r_fn  = _retry or (lambda p: _ag._anthropic_call(model, p, max_tokens=2048))
                rp    = brain._RETRY_PREFIX.format(warning=warn) + prompt if warn else prompt
                return r_fn(rp)
            return full_trace, total_tokens

        monitored.__name__ = f"brain_thinking({model})"
        monitored._brain   = brain
        return monitored


# ── utility ───────────────────────────────────────────────────────────────────

class _suppress_stdout:
    """Context manager: silence stdout from exec'd code."""
    def __enter__(self):
        import contextlib
        self._cm = contextlib.redirect_stdout(io.StringIO())
        self._cm.__enter__()
    def __exit__(self, *a):
        self._cm.__exit__(*a)
