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

        n_fail = sum(1 for i in top if runs[i].label == 0)
        p_fail = n_fail / k

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


def _build_traj_warning(p_fail: float, best_run: _StoredRun | None) -> str:
    lines = [f"[BRAIN — P(fail)={p_fail:.0%}]  Trajectory resembles past failures."]
    if best_run and best_run.metadata:
        lines.append(f"  Most similar failed run: {best_run.metadata[:200]}")
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

        # Two stores: trajectory-level and code-snippet-level
        self._traj_store = TrajectoryStore(embedder, k=k, threshold=threshold)
        self._code_store = FailureStore(embedder, k=k, threshold=threshold)

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

        # Trajectory visualization
        self._trajectory:   list[TrajectoryPoint] = []
        self._traj_lock     = threading.Lock()
        self._task_probe_fn: Callable | None      = None
        self._current_task:  int                  = 0
        self._current_turn:  int                  = 0
        self._passing_codes: list[tuple[str, str]] = []

    # ── task registration ─────────────────────────────────────────────────────

    def set_task(self, task_idx: int, probe_fn: Callable | None = None) -> None:
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
        """Intercept every tool execution. No regex — receives raw input dict.

        Returns a modified result string (original result + brain warning) when
        the brain decides to intervene, or None to leave the result unchanged.

        Fires at most 2 times per task to avoid warning fatigue.
        """
        if name != "python_exec":
            return None
        if self._code_interventions >= 2:
            return None

        code = (input_dict or {}).get("code", "") if isinstance(input_dict, dict) else ""
        if not code:
            return None

        # 1. Deterministic probe tests on the actual code
        probe_fails = self._run_probe(code)

        # 2. kNN over past code snippets
        p_fail, knn_warning = self._code_store.query(code[:2000])

        fired = bool(probe_fails) or (
            p_fail is not None and p_fail >= self._traj_store._threshold
        )

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

        warning = self._build_code_intervention(probe_fails, p_fail, knn_warning)
        self._last_code_warning  = warning
        self.last_p_fail         = p_fail
        self._code_interventions += 1

        pf_str = f"{p_fail:.0%}" if p_fail is not None else "?"
        return (
            f"{result}\n\n"
            f"[BRAIN — P(fail)={pf_str}]\n"
            f"{warning}\n"
            f"Fix the issue above before continuing."
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
            parts.append("Fix the specific issue above — do not rewrite the whole function.")
        elif knn_warning:
            for line in knn_warning.splitlines():
                line = line.strip()
                if line and not line.startswith("[BRAIN") and not line.startswith("Avoid"):
                    parts.append(f"Warning (similar past failure): {line[:160]}")
                    break
        return "\n".join(parts) if parts else "This pattern has caused failures before — double-check correctness."

    # ── storage ───────────────────────────────────────────────────────────────

    def store(self, trace: str, label: int, metadata: str = "") -> None:
        """Store a completed run's full trace as a labeled trajectory.

        Call after every task. The trace is chunked and embedded; both the
        trajectory store (prefix-kNN + Markov) and the buffer are updated.
        """
        self._traj_store.add(trace, label, metadata)
        self._pending_trace = trace

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
        """
        with self._lock:
            buf = self._buffer
        if len(buf) < self._min_chars:
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
        from agents import streaming_agent
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
        import agents as _ag
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
