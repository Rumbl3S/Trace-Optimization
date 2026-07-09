"""brain.py — Latent-trajectory risk prediction for any LLM agent.

ARCHITECTURE
────────────
Each agent run is modeled as a trajectory through a latent reasoning state space.
Every reasoning chunk becomes a LatentStep: an embedding vector plus structural
features derived from that position in the trajectory.

  z₀ → z₁ → z₂ → z₃ → z₄

Each zᵢ carries:
  • embedding   — semantic content (384-dim, sentence-transformers or OpenAI)
  • drift       — 1 − cosine_sim(zᵢ, zᵢ₋₁): how sharply the reasoning turned
  • step_type   — 'reason' | 'code' | 'tool' | 'correct' | 'conclude'
  • index       — position in trajectory

Three signals combine into P(fail):

1. TRAJECTORY FINGERPRINT KNN
   Match trajectories by SHAPE, not just content. The fingerprint captures:
   drift profile (mean, peak, variance), self-correction count, step-type
   ratios, and trajectory length. Two agents can reason about completely
   different topics but follow the same failure pattern — both self-correct
   at step 3, both produce low-confidence conclusions — and this catches it.
   Combined 60/40 with prefix-embedding similarity.

2. MARKOV STATE FAILURE RATE  (activates once ≥ 5 trajectories stored)
   k-means partitions the latent space into thought-state regions. Each region
   accumulates a failure rate from all stored runs that visited it.
   Captures: "this region of reasoning space historically precedes failures."

3. INTERVENTION REASONING (InterventionReasoner)
   When P(fail) ≥ threshold, a fast LLM call generates a specific explanation:
   not "trajectory resembles past failures" but "your ** implementation is
   LEFT-associative (while loop) — 2**3**2 will be 64 not 512. Fix: right-
   recurse: return base ** parse_power(tokens, pos)."

CODE-SNIPPET STORE (on_tool_call)
──────────────────────────────────
python_exec calls also hit a FailureStore keyed by code snippet. Probe tests
run deterministically and kNN over past code catches structural bugs fast.

TASK PROMPT STORE
─────────────────
Prompts (task descriptions) are stored alongside traces so future tasks can
be matched by what was ASKED, not just how the agent responded. Pre-generation
queries compare the new prompt to stored prompts for fast task-level similarity.

PUBLIC API (unchanged from prior versions)
──────────
  brain = BrainAgent(embedder)
  brain.set_task(idx, probe_fn=fn)
  brain.on_tool_call(name, inp, result)   → str | None
  brain.store(trace, label, meta)
  brain.store_code(code, label, meta)
  brain.get_trajectory()                  → list[TrajectoryPoint]
  brain.wrap_any(model_or_callable)       → callable
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


CHUNK_CHARS = 400


# ── Step classification ───────────────────────────────────────────────────────

_CODE_KW = frozenset(['def ', 'class ', '```python', '```py', 'import ', 'lambda '])
_TOOL_KW = frozenset(['[tool:', 'python_exec(', 'tool_result', 'tool_use'])
_CORRECT_KW = frozenset([
    'wait, ', 'wait—', 'wait —', 'actually,', 'actually ', 'i made an error',
    'i made a mistake', 'that\'s wrong', "i'm wrong", 'let me reconsider',
    'i was wrong', 'let me fix', 'i need to fix', 'my mistake', 'let me correct',
    "that won't work", 'this is wrong', 'let me rethink', 'oops,', 'hold on,',
    'no, wait', 'incorrect,',
])
_CONCLUDE_KW = frozenset([
    'answer:', 'therefore,', 'in conclusion', 'the answer is', 'final answer',
    'result:', 'so the answer', 'the result is', 'to summarize', 'in summary',
])


def _classify_step(text: str) -> str:
    """Classify a reasoning chunk into one of 5 latent step types.

    Order matters: tool > correction > code > conclusion > reasoning.
    Corrections are the most predictive failure signal — a self-correction
    event means the agent already produced something wrong.
    """
    t = text.lower()
    if any(kw in t for kw in _TOOL_KW):
        return 'tool'
    if any(kw in t for kw in _CORRECT_KW):
        return 'correct'
    if any(kw in t for kw in _CODE_KW):
        return 'code'
    if any(kw in t for kw in _CONCLUDE_KW):
        return 'conclude'
    return 'reason'


# ── Latent state representation ───────────────────────────────────────────────

@dataclass
class LatentStep:
    """One reasoning step encoded as a latent vector with structural features."""
    text:      str          # raw chunk, truncated to 300 chars
    vec:       np.ndarray   # L2-normalized embedding
    drift:     float        # 1 − cosine_sim(vec, prev_vec): 0=same dir, 2=opposite
    step_type: str          # 'reason' | 'code' | 'tool' | 'correct' | 'conclude'
    index:     int          # position in trajectory


@dataclass
class LatentTrajectory:
    """An ordered sequence of LatentSteps representing one agent run.

    This is the core unit stored in TrajectoryStore. Two trajectories are
    similar if they have similar SHAPES (drift profiles, step-type sequences)
    AND similar semantic content.
    """
    trace_id: str
    label:    int            # 1 = pass, 0 = fail, -1 = in-progress
    steps:    list           # list[LatentStep]
    metadata: str = ""
    task:     str = ""       # task prompt (for display in interventions)

    def prefix_mean_vec(self, n: int = 5) -> np.ndarray | None:
        """Mean embedding of the first n steps (L2-normalized)."""
        s = self.steps[:n]
        if not s:
            return None
        m = np.stack([x.vec for x in s]).mean(axis=0)
        norm = np.linalg.norm(m)
        return m / (norm + 1e-9) if norm > 1e-9 else m

    @property
    def correction_count(self) -> int:
        return sum(1 for s in self.steps if s.step_type == 'correct')

    @property
    def max_drift_step(self) -> int:
        """Index of the step with the highest semantic drift."""
        if not self.steps:
            return 0
        return max(range(len(self.steps)), key=lambda i: self.steps[i].drift)

    def fingerprint(self) -> np.ndarray:
        """10-dim shape descriptor capturing trajectory structure, not content.

        Two agents reasoning about completely different topics can share the
        same fingerprint if they follow the same failure pattern. This is the
        signal that semantic embeddings alone cannot capture.
        """
        if not self.steps:
            return np.zeros(10, dtype='float32')
        drifts = [s.drift for s in self.steps]
        types  = [s.step_type for s in self.steps]
        n      = len(self.steps)
        return np.array([
            float(np.mean(drifts)),                               # average drift
            float(np.max(drifts)),                                # peak drift
            float(np.std(drifts)) if len(drifts) > 1 else 0.0,  # drift variance
            self.correction_count / max(1, n),                   # correction rate
            sum(1 for t in types if t == 'code')    / max(1, n),
            sum(1 for t in types if t == 'reason')  / max(1, n),
            sum(1 for t in types if t == 'tool')    / max(1, n),
            min(n / 20.0, 1.0),                                  # length (capped)
            float(drifts[0]),                                    # first-step drift
            float(drifts[-1]),                                   # final-step drift
        ], dtype='float32')


# ── Motif extraction from failed traces ──────────────────────────────────────────

_EXTRACT_MOTIF_PROMPT = """\
A Python implementation just failed. Extract a REUSABLE failure motif as JSON.

TASK DESCRIPTION:
{task}

FAILED CODE:
```python
{code}
```

FAILURE REASON:
{reason}

Output ONLY a JSON object with these fields (no markdown, no explanation):
{{
  "id": "short_snake_case_id_max_4_words",
  "name": "Human-readable name 5-8 words",
  "description": "One sentence: what logical principle is violated",
  "surface_pattern": "Abstract Python regex matching the STRUCTURAL mistake — see rules below",
  "neg_pattern": "Abstract Python regex matching the CORRECT form; empty string if none",
  "task_keywords": ["2-4 words from the task prompt that scope when this error applies"],
  "confidence": 0.85,
  "recommendation": "Brief instruction fixing the structural mistake (1 line)"
}}

CRITICAL RULES FOR surface_pattern:
- Replace ALL variable names with \\w+ (never use specific names like 'users', 'items')
- Replace ALL string field names with ['\"]\\w+['\"] (never use 'score', 'price', 'email')
- Match both sorted(...) and list.sort(...) forms if the bug applies to both
- The pattern must match ANY code with this structural mistake, not just this specific code
- Do NOT anchor to specific keywords or values that appear in this code by coincidence
- Test mentally: would your pattern match if the dev used different variable names? It must.
- surface_pattern must still match the failed code above (sanity check)
- confidence: 0.80-0.92 only
- Output ONLY the JSON object, nothing else
"""

_MERGE_MOTIF_PROMPT = """\
Two Python implementations failed with the same type of bug. Generalize their motif.

EXISTING MOTIF:
  description: {description}
  current_pattern: {old_pattern}

FIRST FAILURE CODE:
```python
{code1}
```

SECOND FAILURE CODE:
```python
{code2}
```

Write a SINGLE abstract regex that matches BOTH failures and generalizes to other code with this same structural mistake.

Rules:
- Use \\w+ for ALL variable names, never hardcode specific names
- Use ['\"]\\w+['\"] for ALL string field/key names
- Match both sorted() and .sort() if applicable
- The merged pattern must match both code snippets above
- Output ONLY JSON: {{"surface_pattern": "...", "neg_pattern": "..."}}
"""


def _extract_code_from_trace(trace: str) -> str:
    """Extract the last python_exec code block from an agent trace."""
    import json as _json
    last_code = ""
    idx = 0
    while True:
        start = trace.find("[tool:python_exec(", idx)
        if start < 0:
            break
        brace = trace.find("{", start)
        if brace < 0:
            break
        depth = 0
        end = brace
        for i, ch in enumerate(trace[brace:]):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = brace + i + 1
                    break
        try:
            d = _json.loads(trace[brace:end])
            code = d.get("code", "")
            if code:
                last_code = code
        except Exception:
            pass
        idx = end
    return last_code


# ── Intervention reasoning ─────────────────────────────────────────────────────

class InterventionReasoner:
    """Background reasoning agent that generates specific causal explanations.

    When the risk predictor fires, this class:
    1. Analyzes the live trajectory vs the nearest stored failure trajectory
    2. Identifies the divergence point (where the failure pattern began)
    3. Calls haiku to generate a specific, actionable intervention

    The explanation names the exact mistake and the exact fix — not generic
    "reconsider your approach" but "your ** is LEFT-associative; fix: right-
    recurse: return base ** parse_power(tokens, pos)."

    Falls back to a structured (no-LLM) explanation if the LLM call fails
    or times out, so the system always produces a useful intervention.
    """

    _PROMPT = (
        "You are a code debugging advisor. An agent wrote code that is at high risk of failure.\n\n"
        "LIVE CODE (what the agent just executed):\n"
        "```python\n{live_code}\n```\n\n"
        "EXECUTION RESULT: {exec_result}\n\n"
        "SIMILAR PAST FAILURE (a previous task that produced an error like this):\n"
        "  {past_meta}\n\n"
        "SIMILAR PAST SUCCESS (a previous task that worked — model the fix on this):\n"
        "  {pass_meta}\n\n"
        "Write exactly 2 lines and nothing else:\n"
        "STOP: [the specific bug in the LIVE CODE — name the exact line, variable, "
        "or pattern that is wrong, referencing the PAST FAILURE]\n"
        "FIX: [the exact correction — show corrected code or the right pattern, "
        "referencing the PAST SUCCESS if it gives a working example]\n"
        "Be concrete. Reference actual variable names and attribute names from the code above."
    )

    def __init__(self, model: str = "claude-haiku-4-5-20251001"):
        self._model = model
        self._lock  = threading.Lock()

    def generate_structured(
        self, p_fail: float,
        live_traj: LatentTrajectory,
        nearest_failures: list[LatentTrajectory],
        nearest_passes: list[LatentTrajectory] | None = None,
        live_code: str = "",
        exec_result: str = "",
    ) -> str:
        """Fast structured warning from stored metadata. Always available."""
        lines: list[str] = []

        if nearest_failures:
            best = nearest_failures[0]
            if best.metadata:
                lines.append(f"STOP: {best.metadata[:280]}")
            elif exec_result.strip():
                lines.append(f"STOP: Execution produced: {exec_result.strip()[:200]}")
        elif exec_result.strip():
            lines.append(f"STOP: Execution produced: {exec_result.strip()[:200]}")

        if nearest_passes:
            best_pass = nearest_passes[0]
            if best_pass.metadata:
                lines.append(f"FIX: Similar working task: {best_pass.metadata[:280]}")
            else:
                code_steps = [s for s in best_pass.steps if s.step_type in ('code', 'tool')]
                if code_steps:
                    lines.append(f"FIX (working pattern): {code_steps[0].text[:250].strip()}")

        if not lines:
            lines.append(f"[TRAJECTORY RISK P(fail)={p_fail:.0%}] — double-check attribute access and type handling.")

        return "\n".join(lines)

    def generate_llm(
        self, p_fail: float,
        live_traj: LatentTrajectory,
        nearest_failures: list[LatentTrajectory],
        nearest_passes: list[LatentTrajectory] | None = None,
        live_code: str = "",
        exec_result: str = "",
    ) -> str | None:
        """LLM-generated intervention. Returns None on failure (graceful fallback)."""
        from .agents import _anthropic_call
        try:
            best      = nearest_failures[0] if nearest_failures else None
            best_pass = (nearest_passes or [None])[0]

            past_meta = (best.metadata[:400] if best and best.metadata
                        else "(no similar failure stored yet)")

            # For pass context: prefer stored metadata, fall back to code from trajectory steps
            if best_pass and best_pass.metadata:
                pass_meta = best_pass.metadata[:400]
            elif best_pass:
                code_steps = [s.text for s in best_pass.steps if s.step_type in ('code', 'tool')]
                pass_meta = code_steps[0][:400] if code_steps else "(no working example stored yet)"
            else:
                pass_meta = "(no working solution stored yet)"

            prompt = self._PROMPT.format(
                live_code=live_code[:600] or "(not available)",
                exec_result=exec_result[:300].strip() or "(no output)",
                past_meta=past_meta,
                pass_meta=pass_meta,
            )

            text, _ = _anthropic_call(self._model, prompt, max_tokens=180)
            return text.strip()
        except Exception:
            return None

    def generate(
        self, p_fail: float,
        live_traj: LatentTrajectory,
        nearest_failures: list[LatentTrajectory],
        nearest_passes: list[LatentTrajectory] | None = None,
        use_llm: bool = True,
        timeout: float = 2.5,
        live_code: str = "",
        exec_result: str = "",
    ) -> str:
        """Generate intervention with STOP (what failed) + FIX (what worked).

        Pulls the nearest failure for the STOP and the nearest pass for the FIX,
        so the agent gets a concrete working example alongside the warning.
        LLM call runs in background thread; structured fallback is instantaneous.
        """
        if not use_llm:
            return self.generate_structured(
                p_fail, live_traj, nearest_failures, nearest_passes,
                live_code=live_code, exec_result=exec_result,
            )

        result: list[str | None] = [None]

        def _run():
            result[0] = self.generate_llm(
                p_fail, live_traj, nearest_failures, nearest_passes,
                live_code=live_code, exec_result=exec_result,
            )

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        t.join(timeout=timeout)

        return result[0] or self.generate_structured(
            p_fail, live_traj, nearest_failures, nearest_passes,
            live_code=live_code, exec_result=exec_result,
        )


# ── Trajectory store ──────────────────────────────────────────────────────────

@dataclass
class _StoredRun:
    """Internal: one stored LatentTrajectory plus precomputed features."""
    traj:     LatentTrajectory
    fp:       np.ndarray     # fingerprint (10-dim)
    fp_norm:  np.ndarray     # L2-normalized fingerprint


class TrajectoryStore:
    """Stores labeled LatentTrajectory objects. Provides combined P(fail).

    Matching uses a combined score: 60% prefix-embedding similarity (semantic)
    + 40% fingerprint similarity (structural shape). Two trajectories match
    when they LOOK the same AND follow the same reasoning pattern. Neither
    alone is sufficient.

    Markov state failure rates activate once ≥ 5 trajectories are stored.
    """

    def __init__(self, embedder: Callable, k: int = 5,
                 threshold: float = 0.45, n_clusters: int = 12):
        self._embedder   = embedder
        self._k          = k
        self._threshold  = threshold
        self._n_clusters = n_clusters
        self._runs:      list[_StoredRun] = []
        self._lock       = threading.Lock()

        self._km:                    object | None = None
        self._fail_rate: np.ndarray | None         = None
        self._all_vecs:  list[np.ndarray]          = []
        self._all_run_ids: list[int]               = []
        self._markov_dirty: bool                   = False

    def _build_trajectory(
        self, trace: str, label: int, metadata: str = "", task: str = ""
    ) -> LatentTrajectory:
        """Chunk trace → embed each chunk → compute per-step features → LatentTrajectory."""
        chunks = _chunk_trace(trace)
        if not chunks:
            return LatentTrajectory(
                trace_id=uuid.uuid4().hex[:8], label=label,
                steps=[], metadata=metadata, task=task,
            )

        raw_vecs = self._embedder([c[:1500] for c in chunks])
        steps: list[LatentStep] = []
        prev_vec: np.ndarray | None = None

        for i, (chunk, raw) in enumerate(zip(chunks, raw_vecs)):
            vec  = np.asarray(raw, dtype='float32')
            norm = np.linalg.norm(vec)
            vec  = vec / (norm + 1e-9) if norm > 1e-9 else vec

            # Semantic drift: how much the reasoning direction changed
            if prev_vec is None:
                drift = 0.0
            else:
                cos_sim = float(np.dot(vec, prev_vec))
                drift   = max(0.0, min(2.0, 1.0 - cos_sim))

            steps.append(LatentStep(
                text=chunk[:300],
                vec=vec,
                drift=drift,
                step_type=_classify_step(chunk),
                index=i,
            ))
            prev_vec = vec

        return LatentTrajectory(
            trace_id=uuid.uuid4().hex[:8],
            label=label,
            steps=steps,
            metadata=metadata,
            task=task,
        )

    def add(self, trace: str, label: int, metadata: str = "", task: str = "") -> LatentTrajectory:
        """Store a completed trace as a labeled LatentTrajectory."""
        traj = self._build_trajectory(trace, label, metadata, task)
        if not traj.steps:
            return traj

        fp      = traj.fingerprint()
        fp_norm = fp / (np.linalg.norm(fp) + 1e-9)
        run_idx = len(self._runs)
        entry   = _StoredRun(traj=traj, fp=fp, fp_norm=fp_norm)

        with self._lock:
            self._runs.append(entry)
            for s in traj.steps:
                self._all_vecs.append(s.vec)
                self._all_run_ids.append(run_idx)
            self._markov_dirty = True

        # Lazily refit Markov chain once enough data exists
        n_runs   = len(self._runs)
        n_chunks = len(self._all_vecs)
        if n_chunks >= self._n_clusters * 3 and n_runs >= 5 and self._markov_dirty:
            self._refit_markov()

        return traj

    def predict(self, partial_trace: str) -> tuple[float | None, str | None]:
        """Predict P(fail) from a partial trace. Returns (p_fail, warning)."""
        p_fail, nearest, _, live_traj = self.predict_with_context(partial_trace)
        if p_fail is None or p_fail < self._threshold or live_traj is None:
            return p_fail, None
        warning = _build_traj_warning(p_fail, nearest[0] if nearest else None)
        return p_fail, warning

    def predict_with_context(
        self, partial_trace: str
    ) -> tuple[float | None, list[LatentTrajectory], list[LatentTrajectory], LatentTrajectory | None]:
        """Predict P(fail) and return live trajectory + nearest failures + nearest passes.

        Returns (p_fail, nearest_failures, nearest_passes, live_traj).
        nearest_failures: similar runs that failed — drives the STOP message.
        nearest_passes:   similar runs that succeeded — drives the FIX message.
        Both together give InterventionReasoner the full STOP+FIX context so the
        agent gets a concrete working example, not just a warning.
        """
        with self._lock:
            runs      = list(self._runs)
            km        = self._km
            fail_rate = self._fail_rate

        if not runs:
            return 0.0, [], [], None

        live_traj = self._build_trajectory(partial_trace, label=-1)
        if not live_traj.steps:
            return None, [], [], live_traj

        # Signal 1: trajectory fingerprint kNN (shape + semantic prefix)
        p_knn, nearest_failures, nearest_passes = self._fingerprint_knn(live_traj, runs)

        # Signal 2: Markov state failure rate at the current reasoning position
        p_markov: float | None = None
        if km is not None and fail_rate is not None:
            last_vec = live_traj.steps[-1].vec.reshape(1, -1)
            try:
                state    = int(km.predict(last_vec)[0])
                p_markov = float(fail_rate[state])
            except Exception:
                pass

        # Combine: Markov (where we are) + fingerprint kNN (how we got here)
        if p_knn is not None and p_markov is not None:
            p_fail = 0.55 * p_markov + 0.45 * p_knn
        elif p_markov is not None:
            p_fail = p_markov
        else:
            p_fail = p_knn

        return p_fail, nearest_failures, nearest_passes, live_traj

    def _fingerprint_knn(
        self, live_traj: LatentTrajectory, runs: list[_StoredRun]
    ) -> tuple[float | None, list[LatentTrajectory], list[LatentTrajectory]]:
        """kNN using combined prefix-embedding + trajectory shape fingerprint.

        Unlike mean-embedding kNN, this matches trajectories by HOW they reason
        (correction events, code/reasoning ratios, drift profile) in addition to
        WHAT they reason about. Two different tasks can share a failure pattern.
        """
        if not live_traj.steps:
            return None, [], []

        L            = len(live_traj.steps)
        live_prefix  = live_traj.prefix_mean_vec(min(L, 5))
        if live_prefix is None:
            return None, [], []

        live_fp     = live_traj.fingerprint()
        live_fp_n   = live_fp / (np.linalg.norm(live_fp) + 1e-9)

        scored: list[tuple[float, _StoredRun]] = []
        for run in runs:
            if not run.traj.steps:
                continue
            run_prefix = run.traj.prefix_mean_vec(min(L, len(run.traj.steps), 5))
            if run_prefix is None:
                continue

            sim_embed = float(np.dot(live_prefix, run_prefix))
            sim_shape = float(np.dot(live_fp_n, run.fp_norm))
            sim       = 0.60 * sim_embed + 0.40 * sim_shape
            scored.append((sim, run))

        if not scored:
            return None, [], []

        scored.sort(reverse=True, key=lambda x: x[0])
        k   = min(self._k, len(scored))
        # Lower threshold to 0.25 so Markov-driven fires always retrieve context
        top = [(sim, r) for sim, r in scored[:k] if sim >= 0.25]
        if not top:
            return 0.0, [], []

        n_fail = sum(1 for _, r in top if r.traj.label == 0)
        p_fail = n_fail / len(top)

        nearest_failures = [r.traj for _, r in top if r.traj.label == 0]
        nearest_passes   = [r.traj for _, r in top if r.traj.label == 1]
        return p_fail, nearest_failures, nearest_passes

    def _refit_markov(self) -> None:
        """Fit k-means over all stored chunk embeddings; compute per-cluster failure rates."""
        try:
            from sklearn.cluster import MiniBatchKMeans
        except ImportError:
            return

        with self._lock:
            all_vecs  = np.stack(self._all_vecs)
            run_ids   = list(self._all_run_ids)
            runs      = list(self._runs)
            self._markov_dirty = False

        k      = min(self._n_clusters, max(2, len(all_vecs) // 3))
        km     = MiniBatchKMeans(n_clusters=k, random_state=42, n_init=3)
        states = km.fit_predict(all_vecs)

        fail_count  = np.zeros(k)
        visit_count = np.zeros(k)
        for chunk_idx, cluster_id in enumerate(states):
            run_label = runs[run_ids[chunk_idx]].traj.label
            fail_count[cluster_id]  += 1 - run_label
            visit_count[cluster_id] += 1

        with np.errstate(divide='ignore', invalid='ignore'):
            fail_rate = np.where(visit_count > 0, fail_count / visit_count, 0.5)

        with self._lock:
            self._km        = km
            self._fail_rate = fail_rate

    @property
    def n(self)      -> int: return len(self._runs)
    @property
    def n_fail(self) -> int: return sum(1 for r in self._runs if r.traj.label == 0)
    @property
    def n_pass(self) -> int: return sum(1 for r in self._runs if r.traj.label == 1)

    def all_vecs(self) -> tuple[np.ndarray | None, list[int]]:
        """First-step embedding per run + labels (for visualization)."""
        with self._lock:
            if not self._runs:
                return None, []
            mat    = np.stack([r.traj.steps[0].vec for r in self._runs
                               if r.traj.steps])
            labels = [r.traj.label for r in self._runs if r.traj.steps]
        return mat, labels


# ── Code-snippet kNN store ────────────────────────────────────────────────────

@dataclass
class _StoredCode:
    trace_id: str
    label:    int
    excerpt:  str
    tail:     str
    vec:      np.ndarray = field(repr=False)
    metadata: str = ""


class FailureStore:
    """kNN store for short text snippets (code or task prompts).

    Used for two purposes:
      1. Code snippet matching (python_exec traces) — detects structural bugs
      2. Task prompt matching — pre-generation check: does this task resemble
         past tasks that failed?

    Both usages benefit from the same architecture: embed text, rank by
    cosine similarity, compute P(fail) from kNN label majority.
    """

    def __init__(self, embedder: Callable, k: int = 5, threshold: float = 0.45):
        self._embedder  = embedder
        self._k         = k
        self._threshold = threshold
        self._items:    list[_StoredCode] = []
        self._matrix:   np.ndarray | None = None
        self._lock      = threading.Lock()

    def add(self, text: str, label: int, metadata: str = "") -> None:
        if not text.strip():
            return
        vec = self._embedder([text[-1500:]])[0]
        entry = _StoredCode(
            trace_id=uuid.uuid4().hex[:8],
            label=label,
            excerpt=text[:300],
            tail=text[-300:],
            vec=np.asarray(vec, dtype='float32'),
            metadata=metadata,
        )
        with self._lock:
            self._items.append(entry)
            self._matrix = np.stack([i.vec for i in self._items])

    def query(self, text: str) -> tuple[float | None, str | None]:
        """Return (p_fail, warning) for this text. Returns (None, None) if no signal."""
        with self._lock:
            if self._matrix is None or not self._items:
                return None, None
            labels = [i.label for i in self._items]
            mat   = self._matrix.copy()
            items = list(self._items)

        vec  = self._embedder([text[-1500:]])[0]
        sims = mat @ vec
        k    = min(self._k, len(items))
        top  = list(np.argpartition(sims, -k)[-k:])

        # Only count neighbours that are actually similar (cosine ≥ 0.40)
        top = [i for i in top if sims[i] >= 0.40]
        if not top:
            return 0.0, None

        n_fail = sum(1 for i in top if items[i].label == 0)
        p_fail = n_fail / len(top)

        if p_fail < self._threshold:
            return p_fail, None

        fail_nbrs = sorted(
            [(items[i], float(sims[i])) for i in top if items[i].label == 0],
            key=lambda x: x[1], reverse=True,
        )
        if not fail_nbrs:
            return p_fail, None

        lines = [f"[BRAIN — P(fail)={p_fail:.0%}]  Similar input previously failed:"]
        seen: set[str] = set()
        for item, _ in fail_nbrs[:2]:
            if item.metadata and item.metadata not in seen:
                seen.add(item.metadata)
                lines.append(f"  • {item.metadata[:180]}")
        if not seen:
            lines.append(f"  Excerpt: \"{fail_nbrs[0][0].excerpt[:160]}\"")
        lines.append("Avoid the same mistake.")
        return p_fail, "\n".join(lines)

    @property
    def n(self) -> int: return len(self._items)


# ── Logical failure store ─────────────────────────────────────────────────────

class LogicalFailureStore:
    """Stores LLM-extracted failure REASONS, not trace embeddings.

    When a task fails, an LLM call extracts the specific logical error and
    stores its embedding. Future runs are checked against these extracted
    reasons, not raw trace text. This is causal detection, not keyword matching.
    """

    _EXTRACT_PROMPT = (
        "An LLM agent failed the following task. Analyze the trace and identify "
        "the specific logical error, missing step, or wrong assumption that caused "
        "the WRONG ANSWER or WRONG IMPLEMENTATION — not execution problems like "
        "empty tool calls, truncation, or stalling.\n\n"
        "IMPORTANT: Only extract domain-specific logic errors. Examples:\n"
        "  - 'Annualization used sqrt(365) instead of sqrt(252) for daily returns'\n"
        "  - 'Tax bracket calculation applied the top rate to the entire income'\n"
        "  - '** was implemented with a while loop (left-assoc) not right-recursion'\n"
        "If the failure is just empty code / truncation / stalling — reply SKIP.\n\n"
        "Task: {task}\n\n"
        "Trace (last 2000 chars):\n{trace}\n\n"
        "Reply in EXACTLY this format:\n"
        "REASON: <one sentence — the specific algorithmic/logical error>\n"
        "SIGNAL: <one sentence — what to watch for in future code to catch this error>"
    )

    def __init__(self, embedder: Callable, threshold: float = 0.45):
        self._embedder  = embedder
        self._threshold = threshold
        self._patterns: list[dict] = []   # {reason, signal, vec, task}
        self._lock      = threading.Lock()

    def extract_and_store(
        self, trace: str, task: str = "",
        model: str = "claude-haiku-4-5-20251001",
    ) -> str | None:
        """Call an LLM to extract the failure reason and store it (background)."""
        from .agents import _anthropic_call
        prompt = self._EXTRACT_PROMPT.format(task=task[:300], trace=trace[-2000:])
        try:
            raw, _ = _anthropic_call(model, prompt, max_tokens=200)
            if raw.strip().upper().startswith("SKIP"):
                return None
            reason = signal = ""
            for line in raw.splitlines():
                if line.startswith("REASON:"):
                    reason = line[len("REASON:"):].strip()
                elif line.startswith("SIGNAL:"):
                    signal = line[len("SIGNAL:"):].strip()
            if not reason or not signal:
                return None
            vec = np.asarray(self._embedder([signal])[0], dtype='float32')
            with self._lock:
                self._patterns.append(
                    {"reason": reason, "signal": signal, "vec": vec, "task": task[:120]}
                )
            return reason
        except Exception:
            return None

    def query(self, text: str) -> tuple[str | None, float | None]:
        with self._lock:
            if not self._patterns:
                return None, None
            patterns = list(self._patterns)
        vec      = np.asarray(self._embedder([text[-1500:]])[0], dtype='float32')
        best_sim, best = max(
            ((float(np.dot(vec, p["vec"])), p) for p in patterns),
            key=lambda x: x[0],
        )
        if best_sim < self._threshold:
            return None, None
        msg = (
            f"Known failure pattern (similarity {best_sim:.0%}):\n"
            f"  WHAT WENT WRONG BEFORE: {best['reason']}\n"
            f"  WATCH FOR: {best['signal']}"
        )
        return msg, best_sim

    @property
    def n_patterns(self) -> int:
        with self._lock:
            return len(self._patterns)


# ── Evidence + motif types ────────────────────────────────────────────────────

_CONCRETE_EVIDENCE_KINDS = frozenset({
    "constraint_violation", "plan_code_mismatch", "known_failure_motif",
    "runtime_error_pattern", "failed_probe", "repeated_stall", "static_code_bug",
    "contradiction",
})

# Kinds and sources that are contextual signals only — never fire triggers.
# Trajectory resemblance is a domain/semantic prior, not specific to THIS code.
_BANNED_FIRE_KINDS = frozenset({
    "trajectory_similarity", "trajectory_knn", "domain_similarity",
    "task_similarity", "p_fail_prior",
})
_BANNED_SOURCES = frozenset({
    "task_similarity", "domain_similarity", "raw_trajectory_knn",
    "trajectory_store", "trajectory_loop", "trajectory_pulse",
})


@dataclass
class BrainEvidence:
    """A single piece of evidence for or against firing an intervention."""
    kind:       str
    confidence: float
    message:    str
    source:     str
    actionable: bool
    data:       dict = field(default_factory=dict)


@dataclass
class FailureMotif:
    """An abstract failure pattern — not tied to any specific task or answer."""
    id:              str
    name:            str
    description:     str
    surface_pattern: str        # regex to find in code (case-insensitive); "" = semantic-only
    neg_pattern:     str        # if this matches, motif does NOT apply; "" = no exclusion
    task_keywords:   list[str]  # task or reasoning must mention ≥1; [] = always applies
    confidence:      float
    recommendation:  str
    source:          str = "seed"


def make_intervention(evidence: list[BrainEvidence]) -> str:
    """Build a specific, actionable STOP/FIX message from concrete evidence."""
    top = sorted(evidence, key=lambda e: e.confidence, reverse=True)[:3]
    bullets = "\n".join(f"  • {e.message}" for e in top)
    recs: list[str] = []
    seen: set[str] = set()
    for e in top:
        r = e.data.get("recommendation", "")
        if r and r not in seen:
            seen.add(r)
            recs.append(f"  → {r}")
    rec_block = "\n".join(recs) if recs else "  → Reconsider the implementation."
    return (
        "STOP: The monitor detected a likely logical failure before execution.\n\n"
        f"Evidence:\n{bullets}\n\n"
        f"Required correction:\n{rec_block}\n\n"
        "Revise the code and explain the specific fix before calling python_exec again."
    )


# ── Motif store ───────────────────────────────────────────────────────────────

class MotifStore:
    """Stores abstract failure motifs — seeded or learned from real failures.

    A motif is a named pattern of logical failure, not a raw similarity score.
    Fires only when the current code matches a motif's surface_pattern regex or
    its semantic signal embedding — AND the task is in scope (task_keywords).
    Does NOT fire from domain/task-level similarity alone.
    """

    def __init__(self, embedder: Callable, logic_store: "LogicalFailureStore"):
        self._embedder = embedder
        self._logic    = logic_store
        self._motifs:   list[FailureMotif]                    = []
        self._sem_vecs: list[tuple[np.ndarray, FailureMotif]] = []
        self._lock      = threading.Lock()

    def add_motif(self, motif: FailureMotif, signal_text: str = "") -> None:
        with self._lock:
            self._motifs.append(motif)
            if signal_text and not motif.surface_pattern:
                vec  = np.asarray(self._embedder([signal_text[:500]])[0], dtype='float32')
                norm = np.linalg.norm(vec)
                if norm > 1e-9:
                    vec = vec / norm
                self._sem_vecs.append((vec, motif))

    def add_learned(self, trace: str, task: str = "", reason: str = "") -> None:
        """Called after a real failure.

        Attempts to extract a structured FailureMotif (regex surface pattern +
        recommendation) via LLM. Falls back to LogicalFailureStore semantic
        embedding if extraction fails or produces an invalid regex.

        Dedup: if a similar motif already exists (description overlap > 60%),
        merges into a broader pattern instead of adding a duplicate.
        """
        code = _extract_code_from_trace(trace)
        motif = self._extract_motif_llm(code, task=task, reason=reason) if code else None
        if motif is not None:
            similar_idx = self._find_similar_motif(motif)
            if similar_idx >= 0:
                merged = self._merge_motifs(
                    self._surface_motifs[similar_idx], motif, code
                )
                if merged is not None:
                    self._surface_motifs[similar_idx] = merged
                    print(f"[BRAIN MOTIF MERGED] {merged.id}: surface={merged.surface_pattern!r}")
                else:
                    # Merge failed — keep both; broader coverage is better than none
                    self.add_motif(motif)
                    print(f"[BRAIN MOTIF LEARNED] {motif.id}: surface={motif.surface_pattern!r}")
            else:
                self.add_motif(motif)
                print(f"[BRAIN MOTIF LEARNED] {motif.id}: surface={motif.surface_pattern!r}")
        # Always keep semantic fallback — it helps with errors not captured by regex
        self._logic.extract_and_store(trace, task=task)

    def _find_similar_motif(self, new_motif: "FailureMotif") -> int:
        """Return index of an existing motif that describes the same bug, or -1."""
        new_words = set(new_motif.description.lower().split())
        for i, existing in enumerate(self._surface_motifs):
            if existing.source != "learned":
                continue  # never merge into seeded motifs
            existing_words = set(existing.description.lower().split())
            union = new_words | existing_words
            if not union:
                continue
            overlap = len(new_words & existing_words) / len(union)
            if overlap >= 0.55:
                return i
        return -1

    def _merge_motifs(
        self, old: "FailureMotif", new: "FailureMotif", new_code: str
    ) -> "FailureMotif | None":
        """Ask haiku to produce one abstract regex covering both old and new failures."""
        import json as _json
        old_code = getattr(old, "_example_code", "")
        if not old_code:
            # No stored code for old motif — fall back to just broadening the old pattern
            old_code = f"# (code that triggered: {old.surface_pattern})"
        try:
            from .agents import _anthropic_call
            prompt = _MERGE_MOTIF_PROMPT.format(
                description=old.description,
                old_pattern=old.surface_pattern,
                code1=old_code[:600],
                code2=new_code[:600],
            )
            text, _ = _anthropic_call("claude-haiku-4-5-20251001", prompt, max_tokens=300)
            text = text.strip()
            if text.startswith("```"):
                text = re.sub(r"^```[a-z]*\n?", "", text)
                text = re.sub(r"\n?```$", "", text)
            data = _json.loads(text)
            surface = data.get("surface_pattern", "")
            neg = data.get("neg_pattern", old.neg_pattern)
            if not surface:
                return None
            re.compile(surface)
            if neg:
                re.compile(neg)
            # Merged pattern must match BOTH code examples
            if not re.search(surface, new_code, re.IGNORECASE):
                return None
            merged = FailureMotif(
                id=old.id,
                name=old.name,
                description=old.description,
                surface_pattern=surface,
                neg_pattern=neg or "",
                task_keywords=list(set(old.task_keywords) | set(new.task_keywords)),
                confidence=max(old.confidence, new.confidence),
                recommendation=old.recommendation,
                source="learned",
            )
            merged._example_code = new_code  # store latest failure code
            return merged
        except Exception:
            return None

    def _extract_motif_llm(
        self, code: str, task: str, reason: str,
    ) -> "FailureMotif | None":
        """Call haiku to extract a structured FailureMotif from a failed code snippet."""
        import json as _json
        try:
            from .agents import _anthropic_call
        except ImportError:
            return None

        prompt = _EXTRACT_MOTIF_PROMPT.format(
            task=task[:400] or "(not provided)",
            code=code[:800],
            reason=reason[:300] or "(not provided)",
        )
        try:
            text, _ = _anthropic_call("claude-haiku-4-5-20251001", prompt, max_tokens=400)
            text = text.strip()
            # Strip markdown fences if present
            if text.startswith("```"):
                text = re.sub(r"^```[a-z]*\n?", "", text)
                text = re.sub(r"\n?```$", "", text)
            data = _json.loads(text)
        except Exception:
            return None

        surface = data.get("surface_pattern", "")
        neg     = data.get("neg_pattern", "")
        if not surface:
            return None
        # Validate regex patterns before using them
        try:
            re.compile(surface)
            if neg:
                re.compile(neg)
        except re.error:
            return None
        # Sanity check: surface pattern must actually match the bad code
        if not re.search(surface, code, re.IGNORECASE):
            return None

        motif = FailureMotif(
            id              = str(data.get("id", "learned"))[:40],
            name            = str(data.get("name", "Learned motif"))[:80],
            description     = str(data.get("description", ""))[:200],
            surface_pattern = surface,
            neg_pattern     = neg,
            task_keywords   = [str(k) for k in data.get("task_keywords", [])[:6]],
            confidence      = max(0.70, min(0.92, float(data.get("confidence", 0.82)))),
            recommendation  = str(data.get("recommendation", ""))[:300],
            source          = "learned",
        )
        motif._example_code = code  # stored for merge dedup
        return motif

    def check(
        self, code: str, task: str, reasoning: list,
        include_learned: bool = True,
    ) -> list[BrainEvidence]:
        """Return evidence for any motif matched in code/task/reasoning.

        include_learned=False skips the LogicalFailureStore semantic query.
        Use False in before_tool_call (predictive) to avoid domain-prior false
        positives: learned signals embed close to ALL code in the same domain,
        not just code that actually contains the mistake.
        Use True in on_tool_call (reactive) where an execution error has already
        confirmed something went wrong in this specific run.
        """
        if not code.strip():
            return []
        task_lower     = task.lower()
        code_lower     = code.lower()
        reasoning_text = " ".join(s.text for s in reasoning).lower()
        evidence: list[BrainEvidence] = []

        with self._lock:
            motifs   = list(self._motifs)
            sem_vecs = list(self._sem_vecs)

        # Surface-pattern motifs: match regex in code, fast and precise
        for motif in motifs:
            if not motif.surface_pattern:
                continue
            if motif.task_keywords and not any(
                kw in task_lower or kw in reasoning_text for kw in motif.task_keywords
            ):
                continue
            # Negative pattern: code already has the safe form → skip
            if motif.neg_pattern and re.search(motif.neg_pattern, code_lower):
                continue
            if re.search(motif.surface_pattern, code_lower):
                evidence.append(BrainEvidence(
                    kind="known_failure_motif",
                    confidence=motif.confidence,
                    message=f"{motif.name}: {motif.description}",
                    source=f"motif:{motif.id}",
                    actionable=True,
                    data={"recommendation": motif.recommendation, "motif_id": motif.id},
                ))

        # Semantic-only motifs (no surface_pattern): embedding similarity
        if sem_vecs:
            try:
                vec  = np.asarray(self._embedder([code[-1500:]])[0], dtype='float32')
                norm = np.linalg.norm(vec)
                if norm > 1e-9:
                    vec = vec / norm
                for mv, motif in sem_vecs:
                    if motif.task_keywords and not any(kw in task_lower for kw in motif.task_keywords):
                        continue
                    sim = float(np.dot(vec, mv))
                    if sim >= 0.75:   # high threshold — semantic only, no surface anchor
                        evidence.append(BrainEvidence(
                            kind="known_failure_motif",
                            confidence=min(0.95, sim),
                            message=f"{motif.name} (semantic {sim:.0%}): {motif.description}",
                            source=f"motif:{motif.id}:semantic",
                            actionable=True,
                            data={"recommendation": motif.recommendation},
                        ))
            except Exception:
                pass

        # Learned motifs (from LogicalFailureStore, LLM-extracted from real failures).
        # ONLY included when include_learned=True. When False (before_tool_call path),
        # these are suppressed because their semantic embeddings co-locate with ALL code
        # in the same domain, causing a domain-prior false positive on every task.
        if include_learned and self._logic.n_patterns > 0:
            logic_msg, p_logic = self._logic.query(code)
            if logic_msg and p_logic is not None and p_logic >= 0.75:
                evidence.append(BrainEvidence(
                    kind="known_failure_motif",
                    confidence=p_logic,
                    message=logic_msg[:300],
                    source="motif_store:learned",
                    actionable=True,
                    data={"recommendation": "See failure description above."},
                ))

        return evidence


# ── Constraint checker ────────────────────────────────────────────────────────

class ConstraintChecker:
    """Fires only when the task prompt EXPLICITLY states a requirement that the
    code violates. Does not fire from domain knowledge alone.
    """

    _DEFS = [
        {
            "keywords": [
                r"population\s+std", r"population\s+standard\s+deviation",
                r"population\s+variance", r"divide\s+by\s+n\b", r"\bn,?\s+not\s+n-1",
                r"ddof\s*=\s*0",
            ],
            "violation_re": r"ddof\s*=\s*1|statistics\.stdev\s*\(",
            "safe_re": r"ddof\s*=\s*0",
            "violation_msg": (
                "Task requires population std (divide by n) but code uses sample std "
                "(ddof=1 or statistics.stdev — divides by n-1)."
            ),
            "recommendation": "np.std(x, ddof=0)  or  sum((x-mu)**2)/n",
        },
        {
            "keywords": [r"\bcompound(?:ed|ing)?\b", r"compound\s+(?:daily|conversion|rate)"],
            "violation_re": r"(rf|risk_free|rate)\s*[^(]*?/\s*252",
            "safe_re": r"\(1\s*\+.*?\)\s*\*\*|\bexp\s*\(",
            "violation_msg": (
                "Task requires compound rate conversion but code uses simple /252 division."
            ),
            "recommendation": "rf_daily = (1 + risk_free_annual)**(1/252) - 1",
        },
        {
            "keywords": [r"geometric\s+annuali", r"compound\s+annuali"],
            "violation_re": r"\bmean\s*\(.*?\)\s*\*\s*252|\*\s*252\b",
            "safe_re": r"\*\*\s*\(\s*252|\bpow\s*\(",
            "violation_msg": (
                "Task requires geometric/compound annualization but code uses arithmetic (mean * 252)."
            ),
            "recommendation": "(1 + total_return)**(252/n) - 1  where total_return=product(1+r)-1",
        },
        {
            "keywords": [r"all\s+(?:periods?|returns?|n\s+periods?)", r"total\s+n\b"],
            "violation_re": r"/\s*len\s*\(.*?(?:neg|below|downside|bad|loss)",
            "safe_re": "",
            "violation_msg": (
                "Task requires dividing by ALL n periods but code divides by count of "
                "below-threshold returns only."
            ),
            "recommendation": "sqrt(sum(min(r-mar,0)**2 for r in returns) / len(returns))",
        },
        {
            "keywords": [r"one-tailed\b", r"one\s+tailed\b"],
            "violation_re": r"\b1\.96\b|\b2\.576\b",
            "safe_re": r"\b1\.6449\b|\b1\.645\b|\b2\.326\b",
            "violation_msg": (
                "Task specifies one-tailed quantile (VaR) but code uses two-tailed "
                "z-scores (1.96 or 2.576). One-tailed 95% = 1.6449, 99% = 2.3263."
            ),
            "recommendation": "z = {0.95: 1.6449, 0.99: 2.3263}[confidence]",
        },
        {
            "keywords": [r"\bito\b", r"ito\s+(?:correction|lemma)", r"unbiased.*gbm", r"gbm.*unbiased"],
            "violation_re": r"\bdrift\s*=\s*mu|\bmu_annual\s*/\s*252",
            "safe_re": r"0\.5\s*\*?\s*sigma|sigma\s*\*?\s*0\.5",
            "violation_msg": (
                "Task requires Ito-corrected GBM drift but code uses drift = mu/dt "
                "without the -0.5*sigma**2 term."
            ),
            "recommendation": "drift = (mu_annual - 0.5 * sigma_annual**2) / 252",
        },
    ]

    def __init__(self):
        self._compiled: list[dict] = []
        for defn in self._DEFS:
            self._compiled.append({
                **defn,
                "_kw_re":   [re.compile(kw, re.IGNORECASE) for kw in defn["keywords"]],
                "_viol_re": re.compile(defn["violation_re"], re.IGNORECASE) if defn["violation_re"] else None,
                "_safe_re": re.compile(defn["safe_re"],      re.IGNORECASE) if defn.get("safe_re") else None,
            })

    def check(self, task: str, code: str) -> list[BrainEvidence]:
        if not code.strip():
            return []
        evidence: list[BrainEvidence] = []
        for defn in self._compiled:
            if not any(p.search(task) for p in defn["_kw_re"]):
                continue
            if defn["_viol_re"] and defn["_viol_re"].search(code):
                if defn["_safe_re"] and defn["_safe_re"].search(code):
                    continue
                evidence.append(BrainEvidence(
                    kind="constraint_violation",
                    confidence=0.90,
                    message=defn["violation_msg"],
                    source="constraint_checker",
                    actionable=True,
                    data={"recommendation": defn["recommendation"]},
                ))
        return evidence


# ── Plan-code mismatch checker ────────────────────────────────────────────────

class PlanCodeMismatchChecker:
    """Fires when the model's stated reasoning contradicts its proposed code."""

    _CHECKS: list[tuple] = [
        (
            re.compile(r"compound(?:ing|ed)?\s+(?:conversion|daily|rate|rf)", re.I),
            re.compile(r"(rf|risk_free|rate)\s*[^(]*?/\s*252", re.I),
            re.compile(r"\(1\s*\+.*?\)\s*\*\*|\bexp\s*\(", re.I),
            "Reasoning mentions compound rate conversion but code uses simple /252.",
            "rf_daily = (1 + risk_free_annual)**(1/252) - 1",
        ),
        (
            re.compile(r"population\s+(?:std|standard\s+deviation|variance)", re.I),
            re.compile(r"ddof\s*=\s*1|statistics\.stdev", re.I),
            None,
            "Reasoning states population std but code uses sample std (ddof=1).",
            "np.std(x, ddof=0) or sum((x-mu)**2)/n",
        ),
        (
            re.compile(r"ito\s+(?:correction|lemma)|geometric\s+brownian", re.I),
            re.compile(r"\bdrift\s*=\s*mu", re.I),
            re.compile(r"0\.5\s*\*?\s*sigma|sigma\s*\*?\s*0\.5", re.I),
            "Reasoning mentions Ito correction but code drift lacks -0.5*sigma**2.",
            "(mu_annual - 0.5 * sigma_annual**2) / 252",
        ),
        (
            re.compile(r"all\s+(?:periods?|returns?|n)", re.I),
            re.compile(r"/\s*len\s*\(.*?(?:neg|below|downside)", re.I),
            None,
            "Reasoning says divide by all periods but code divides by negatives only.",
            "sqrt(sum(min(r-mar,0)**2 for r in returns) / len(returns))",
        ),
        (
            re.compile(r"geometric\s+(?:annuali|compound)", re.I),
            re.compile(r"\bmean\s*\(.*?\)\s*\*\s*252|\*\s*252\b", re.I),
            re.compile(r"\*\*\s*\(\s*252|\bpow\s*\(", re.I),
            "Reasoning mentions geometric annualization but code uses arithmetic (mean*252).",
            "(1 + total_return)**(252/n) - 1",
        ),
    ]

    def check(self, live_steps: list, code: str) -> list[BrainEvidence]:
        if not live_steps or not code.strip():
            return []
        recent = " ".join(s.text for s in live_steps[-6:])
        evidence: list[BrainEvidence] = []
        for plan_re, viol_re, safe_re, message, recommendation in self._CHECKS:
            if not plan_re.search(recent):
                continue
            if not viol_re.search(code):
                continue
            if safe_re and safe_re.search(code):
                continue
            evidence.append(BrainEvidence(
                kind="plan_code_mismatch",
                confidence=0.80,
                message=message,
                source="plan_code_checker",
                actionable=True,
                data={"recommendation": recommendation},
            ))
        return evidence


# ── Helpers ───────────────────────────────────────────────────────────────────


def _chunk_trace(trace: str) -> list[str]:
    """Split a trace into ordered semantic chunks.

    Chunks at tool-call boundaries first, then paragraph breaks, then
    hard-caps at CHUNK_CHARS so embeddings stay well-conditioned.
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
    best_fail: LatentTrajectory | None,
) -> str:
    lines = [f"[BRAIN — P(fail)={p_fail:.0%}]  Trajectory resembles past failures."]
    if best_fail:
        if best_fail.metadata:
            lines.append(f"  Nearest failure: {best_fail.metadata[:200]}")
        fail_pattern = " → ".join(s.step_type for s in best_fail.steps[:6])
        if fail_pattern:
            lines.append(f"  Failure step pattern: {fail_pattern}")
    lines.append("Reconsider your current approach before continuing.")
    return "\n".join(lines)


def _inject_cot(prompt: str) -> str:
    _code_kw = ("function", "class", "implement", "write a", "def ", "code")
    is_code  = any(kw in prompt.lower() for kw in _code_kw)
    suffix   = (
        "\n\nWork through your approach step by step, explaining your reasoning. "
        "Then provide the complete implementation in a ```python code block."
        if is_code else
        "\n\nWork through this step by step, showing each reasoning step. "
        "Give your final answer as 'ANSWER: ...'."
    )
    return prompt + suffix


# ── Data types for trajectory visualization ───────────────────────────────────

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


# ── BrainAgent ────────────────────────────────────────────────────────────────

class BrainAgent:
    """Inference-time failure detector using latent trajectory analysis.

    Wraps any agent or model. Monitors reasoning in real time and intervenes
    with specific, causal explanations when the trajectory enters a region
    historically associated with failure.

    Usage::
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
        "STOP. The trajectory monitor detected a high-risk reasoning pattern.\n\n"
        "{warning}\n\n"
        "Apply the specific fix above. "
        "Write the correct, complete implementation.\n\n"
        "Task:\n"
    )

    def __init__(
        self, embedder: Callable, k: int = 5, threshold: float = 0.45,
        check_interval: float = 1.0, min_chars: int = 200,
    ):
        self._embedder       = embedder
        self._check_interval = check_interval
        self._min_chars      = min_chars

        self._traj_store  = TrajectoryStore(embedder, k=k, threshold=threshold)
        self._code_store  = FailureStore(embedder, k=k, threshold=threshold)
        self._task_store  = FailureStore(embedder, k=k, threshold=threshold)
        self._logic_store = LogicalFailureStore(embedder)
        self._reasoner    = InterventionReasoner()

        self._motif_store         = MotifStore(embedder, self._logic_store)
        self._constraint_checker  = ConstraintChecker()
        self._plan_code_checker   = PlanCodeMismatchChecker()
        self._fire_threshold: float = threshold

        # Error-pattern store: (embedding, error_message) pairs from past failures.
        # Separate from FailureStore because all entries have label=0 (failures only)
        # and we only need cosine similarity, not majority-vote p_fail.
        self._error_patterns: list[tuple[np.ndarray, str]] = []
        self._error_lock = threading.Lock()

        # Live reasoning trajectory: LatentSteps built from push() calls.
        # Captures the agent's thinking between tool calls, not just the tool calls.
        self._live_steps: list[LatentStep] = []

        self._current_task_str: str = ""

        # Streaming buffer
        self._buffer  = ""
        self._lock    = threading.Lock()
        self._bail_ev = threading.Event()
        self._stop_ev = threading.Event()
        self._thread: threading.Thread | None = None

        # Run state
        self.last_p_fail:         float | None = None
        self.last_warning:        str   | None = None
        self._pending_trace:      str          = ""
        self._last_code_warning:  str          = ""
        self._code_interventions: int          = 0

        self._turn_count:   int      = 0
        self._fire_turn:    int | None = None
        self._stall_streak: int      = 0
        self._stall_threshold: int   = 2
        self._stall_fired:  bool     = False

        # Visualization (reset between tasks)
        self._trajectory:   list[TrajectoryPoint] = []
        self._traj_lock     = threading.Lock()
        self._task_probe_fn: Callable | None      = None
        self._current_task:  int                  = 0
        self._current_turn:  int                  = 0
        self._passing_codes: list[tuple[str, str]] = []

        # Fire audit — set when a hook returns a STOP message, cleared by reset()
        self.last_fire: dict | None = None

    # ── task registration ─────────────────────────────────────────────────────

    def set_task(self, task_idx: int, probe_fn: Callable | None = None,
                 task: str = "") -> None:
        """Register the current task and optional deterministic probe."""
        self._current_task       = task_idx
        self._current_turn       = 0
        self._task_probe_fn      = probe_fn
        self._code_interventions = 0
        if task:
            self._current_task_str = task

    # ── pre-generation risk check ─────────────────────────────────────────────

    def predict_pre_generation(
        self, prompt: str, use_llm: bool = True
    ) -> tuple[float | None, str | None]:
        """Check P(fail) before generating. Compares the prompt against:
          1. The trajectory store (populated by seeds + past runs as trajectories)
          2. The task store (populated by past task prompts via store()/seed())

        The trajectory store comparison works with seeds because seed traces and
        task prompts co-locate in embedding space — both describe the same domain
        (e.g., "** right-recursion" in code and in English descriptions).

        Returns (p_fail, specific_warning). The warning names the exact bug and
        fix when the InterventionReasoner generates a successful explanation.
        Falls back to a structured warning derived from stored metadata.
        """
        # Signal 1: trajectory store (works with seeds, grows with each teach())
        p_traj, nearest_failures, nearest_passes, live_traj = (
            self._traj_store.predict_with_context(prompt)
        )

        # Signal 2: task prompt store (grows with each teach())
        p_task, task_warning = self._task_store.query(prompt)

        # Take the stronger signal
        if p_traj is not None and p_task is not None:
            p_fail = max(p_traj, p_task)
        elif p_traj is not None:
            p_fail = p_traj
        elif p_task is not None:
            p_fail = p_task
        else:
            return None, None

        if p_fail < self._traj_store._threshold:
            return p_fail, None

        # Reasoner gets failures (STOP) AND passes (FIX) for a complete intervention
        if live_traj is not None and live_traj.steps and nearest_failures:
            specific = self._reasoner.generate(
                p_fail, live_traj, nearest_failures,
                nearest_passes=nearest_passes, use_llm=use_llm,
            )
            return p_fail, specific

        # Fallback: structured warning from task store
        return p_fail, task_warning

    # ── should_fire / make_intervention ──────────────────────────────────────

    def should_fire(self, evidence: list[BrainEvidence]) -> tuple[bool, str]:
        """Central fire gate. Returns (True, message) only when there is at least
        one piece of named, concrete, actionable evidence from a non-banned source.

        p_fail / trajectory resemblance / domain similarity NEVER cause a fire,
        regardless of their magnitude. They may only be used as metadata.

        A fire requires the brain to name a specific mistake:
          "This code uses rf/252 (additive) where compounding is required."
          "ddof=1 violates the explicit population-std requirement in the task."
        Not:
          "Trajectory resembles past failures."
          "p_fail is high."
        """
        concrete = [
            e for e in evidence
            if e.actionable
            and e.kind in _CONCRETE_EVIDENCE_KINDS
            and e.kind not in _BANNED_FIRE_KINDS
            and e.source.split(":")[0] not in _BANNED_SOURCES
            and len(e.message) >= 20
            and e.confidence >= self._fire_threshold
        ]
        if not concrete:
            return False, ""
        return True, make_intervention(concrete)

    # ── seed_motifs ───────────────────────────────────────────────────────────

    def seed_motifs(self, motif_dicts: list[dict]) -> None:
        """Pre-populate MotifStore with abstract failure pattern dicts.

        Each dict must have: id, surface_pattern, task_keywords, confidence.
        Optional: neg_pattern, name, description, recommendation.
        """
        for d in motif_dicts:
            motif = FailureMotif(
                id              = d["id"],
                name            = d.get("name", d["id"].replace("_", " ")),
                description     = d.get("description", d.get("recommendation", d["id"])),
                surface_pattern = d.get("surface_pattern", ""),
                neg_pattern     = d.get("neg_pattern", ""),
                task_keywords   = d.get("task_keywords", []),
                confidence      = d["confidence"],
                recommendation  = d.get("recommendation", ""),
                source          = "seed",
            )
            self._motif_store.add_motif(motif)

    # ── before_tool_call: pre-execution prediction ────────────────────────────

    def before_tool_call(self, name: str, input_dict: dict) -> str | None:
        """Called BEFORE the tool executes. Fires only on concrete current evidence.

        Architecture (per spec):
          p_fail from trajectory kNN = weak context, NEVER a trigger alone.
          Specific logical evidence   = required trigger.
          Combined score              = max(specific) * 0.75 + p_fail * 0.25

        Fire sources (all require direct inspection of THIS code, not domain priors):
          - ConstraintChecker       : task explicitly says X, code violates X
          - PlanCodeMismatchChecker : reasoning says X, code does Y
          - MotifStore (surface)    : code matches a seeded regex failure pattern

        Learned semantic motifs (LogicalFailureStore) are EXCLUDED here because their
        embeddings co-locate with ALL code in the same domain after the first failure
        is stored. They're used reactively in on_tool_call instead.
        """
        if name != "python_exec" or self._code_interventions >= 2:
            return None

        code = (input_dict or {}).get("code", "")
        if not code.strip():
            return None

        # Collect specific, current-code evidence only — no learned semantic signals
        evidence: list[BrainEvidence] = []
        evidence += self._constraint_checker.check(self._current_task_str, code)
        evidence += self._plan_code_checker.check(self._live_steps, code)
        evidence += self._motif_store.check(
            code=code, task=self._current_task_str, reasoning=self._live_steps,
            include_learned=False,   # no domain-prior semantic signals here
        )

        # Gate: must have at least one specific, actionable, high-confidence piece
        # of evidence that names a concrete mistake in THIS code.
        has_specific = any(
            e.actionable
            and e.confidence >= 0.70
            and e.kind in _CONCRETE_EVIDENCE_KINDS
            and e.source not in _BANNED_SOURCES
            for e in evidence
        )
        if not has_specific:
            return None

        # p_fail from trajectory kNN: amplifies but does not trigger
        p_traj: float = 0.0
        if self._live_steps and len(self._traj_store._runs) >= 3:
            p_knn, _, _, _ = self._traj_store.predict_with_context(
                "\n".join(s.text for s in self._live_steps)
            )
            if p_knn is not None:
                p_traj = p_knn

        max_conf = max(e.confidence for e in evidence if e.kind in _CONCRETE_EVIDENCE_KINDS)
        combined = max_conf * 0.75 + p_traj * 0.25

        if combined < self._fire_threshold:
            return None

        assert evidence, "before_tool_call: cannot fire with empty evidence list"
        msg = make_intervention(evidence)
        self.last_p_fail         = combined
        self.last_warning        = msg
        self.last_fire           = {
            "task": self._current_task,
            "hook": "before_tool_call",
            "p_traj": p_traj,
            "combined": combined,
            "evidence": [(e.kind, round(e.confidence, 3), e.message[:120]) for e in evidence],
        }
        print("[BRAIN FIRE]", {
            "task": self._current_task,
            "hook": "before_tool_call",
            "p_traj": round(p_traj, 3),
            "combined": round(combined, 3),
            "evidence": [(e.kind, round(e.confidence, 3), e.message[:80]) for e in evidence],
            "msg_head": msg[:100],
        })
        self._code_interventions += 1
        pt = TrajectoryPoint(
            task_idx=self._current_task, turn=self._current_turn,
            p_fail=combined, probe_fails=[], fired=True,
        )
        with self._traj_lock:
            self._trajectory.append(pt)
        return f"⚠️ BRAIN:\n{msg}"

    # ── on_tool_call: reactive post-execution hook ────────────────────────────

    def on_tool_call(self, name: str, input_dict: dict, result: str) -> str | None:
        """Post-execution hook. Called AFTER the tool runs (or after before_tool_call
        blocked and returned a warning). Reactive signals only — the predictive
        signals (trajectory risk, logical gap) live in before_tool_call.

        Returns a modified result string when intervening, None otherwise.
        Fires at most 2 times per task total (shared counter with before_tool_call).

        Signals:
          1. Stall detector    — consecutive empty/no-output calls
          2. Probe tests       — deterministic unit tests (python_exec only)
          3. Logical failure   — LLM-extracted reasons from past failures
          4. Code snippet kNN  — past code similarity (python_exec only)
          5. Trajectory kNN    — partial trace vs past run prefixes
        """
        if self._code_interventions >= 2:
            return None

        self._turn_count += 1
        is_no_output = not result or result.strip() in ("", "(no output)", "None")

        code = ""
        if name == "python_exec":
            code = (input_dict or {}).get("code", "") if isinstance(input_dict, dict) else ""

        # ── Stall detector ────────────────────────────────────────────────────
        no_input = (not code if name == "python_exec" else
                    not any(str(v).strip() for v in (input_dict or {}).values()))
        if no_input or is_no_output:
            self._stall_streak += 1
            if self._stall_streak >= self._stall_threshold and not self._stall_fired:
                self._stall_fired        = True
                self._code_interventions += 1
                if self._fire_turn is None:
                    self._fire_turn = self._turn_count
                msg = (
                    f"[BRAIN — STALL after {self._stall_streak} unproductive calls]\n"
                    f"Your last {self._stall_streak} '{name}' calls produced no meaningful "
                    f"output. Stop and try a fundamentally different approach.\n"
                    f"  1. Do not repeat the same call with the same or empty inputs.\n"
                    f"  2. Produce one complete, self-contained implementation.\n"
                    f"Make your next call count."
                )
                self._last_code_warning = msg
                self.last_warning       = msg
                return f"{result}\n\n{msg}" if result else msg
            return None
        else:
            self._stall_streak = 0

        # Accumulate partial trace
        input_summary = code if name == "python_exec" else str(input_dict or "")[:500]
        self._pending_trace += f"\n[tool:{name}]\n{input_summary}\n[result]\n{result[:1000]}\n"

        # ── Probe tests (deterministic, python_exec only) ─────────────────────
        probe_fails: list[str] = []
        if name == "python_exec" and code:
            probe_fails = self._run_probe(code)

        # ── Execution error: did this code actually fail? ──────────────────────
        exec_error = ""
        if name == "python_exec" and result:
            # Extract the specific error line from the execution result
            for line in reversed(result.strip().splitlines()):
                s = line.strip()
                if s and any(s.startswith(kw) for kw in (
                    "Error", "Exception", "Traceback", "TypeError", "AttributeError",
                    "ValueError", "KeyError", "NameError", "IndexError", "AssertionError",
                )):
                    exec_error = s
                    break
            if not exec_error and any(kw in result for kw in ("Error", "Exception")):
                # Short result that is entirely an error
                stripped = result.strip()
                if len(stripped) < 300:
                    exec_error = stripped.splitlines()[-1].strip()

        # ── Reactive: did execution produce the same error as a past failure? ───
        p_err:       float | None = None
        err_context: str   | None = None
        if exec_error:
            with self._error_lock:
                patterns = list(self._error_patterns)
            if patterns:
                qvec = np.asarray(self._embedder([exec_error])[0], dtype='float32')
                sims = [float(np.dot(qvec, pv)) for pv, _ in patterns]
                best_sim = max(sims)
                if best_sim >= 0.55:
                    best_idx    = int(np.argmax(sims))
                    p_err       = best_sim
                    err_context = patterns[best_idx][1]

        # ── Reactive logical gap: exec error + learned failure pattern ───────────
        # Only queried when we ALREADY have an execution error — prevents the
        # LogicalFailureStore from acting as a domain prior (it embeds near all
        # code in the same domain once one failure is stored, not just wrong code).
        # Threshold raised to 0.75 (vs old 0.62) for the same reason.
        p_logic:       float | None = None
        logic_warning: str   | None = None
        if code and exec_error and self._logic_store.n_patterns > 0:
            logic_warning, p_logic = self._logic_store.query(code)

        # ── Trajectory kNN: context only (nearest failures + passes for STOP/FIX)
        p_traj, nearest_failures, nearest_passes, live_traj = (
            self._traj_store.predict_with_context(self._pending_trace)
            if len(self._traj_store._runs) >= 3
            else (None, [], [], None)
        )

        # p_fail: best available signal, for display / logging only
        p_fail: float | None = (
            p_err if p_err is not None else
            p_logic if p_logic is not None else
            p_traj
        )

        # ── Fire decision ──────────────────────────────────────────────────────
        # Reactive signals only — each requires THIS execution to have produced
        # specific evidence (error, probe fail). Domain/trajectory priors alone
        # never trigger a fire here; pre-execution motif detection lives in
        # before_tool_call instead.
        if self._task_probe_fn is not None and name == "python_exec":
            fired = bool(probe_fails)
        else:
            fired = bool(probe_fails) or (
                exec_error != "" and p_err is not None and p_err >= 0.45
            ) or (
                # Logical gap: only fires if code already errored AND signal is
                # highly specific (0.75, not 0.62) to avoid domain-prior fires
                exec_error != "" and logic_warning is not None
                and p_logic is not None and p_logic >= 0.75
            )

        pt = TrajectoryPoint(
            task_idx=self._current_task, turn=self._current_turn,
            p_fail=p_fail, probe_fails=probe_fails, fired=fired,
        )
        with self._traj_lock:
            self._trajectory.append(pt)
        self._current_turn += 1

        if not fired:
            return None

        if self._fire_turn is None:
            self._fire_turn = self._turn_count

        # ── Generate STOP/FIX from concrete evidence only ─────────────────────
        # Warning must name the SPECIFIC mistake — not trajectory resemblance.
        # Priority order:
        #   1. Probe failures (deterministic, most specific)
        #   2. Execution error that matches a known failure pattern
        #   3. Logical gap confirmed by code + error (logic_warning + exec_error)
        # Trajectory context (live_traj, nearest_failures) is available for the
        # LLM reasoner ONLY if there is already a concrete error to anchor on.
        # Never generate "Trajectory resembles past failures" as the primary message.
        if probe_fails:
            warning = self._build_code_intervention(probe_fails, p_fail, None)
        elif exec_error:
            exec_or_logic = logic_warning or exec_error
            if live_traj is not None and live_traj.steps and nearest_failures:
                # LLM reasoner anchored to the ACTUAL error (not trajectory alone)
                warning = self._reasoner.generate(
                    p_fail or 0.5, live_traj, nearest_failures,
                    nearest_passes=nearest_passes, use_llm=True, timeout=2.0,
                    live_code=code[:800], exec_result=exec_or_logic[:400],
                )
            else:
                stop = f"STOP: Execution error: {exec_error[:200]}"
                fix  = (f"FIX: {nearest_passes[0].metadata[:220]}"
                        if nearest_passes and nearest_passes[0].metadata
                        else "FIX: review your implementation logic.")
                warning = f"{stop}\n{fix}"
        elif logic_warning:
            fix = (f"FIX: {nearest_passes[0].metadata[:220]}"
                   if nearest_passes and nearest_passes[0].metadata
                   else "FIX: review attribute access, return types, and type assumptions.")
            warning = f"STOP: {logic_warning}\n{fix}"
        else:
            warning = "STOP: check your implementation."

        self._last_code_warning  = warning
        self.last_warning        = warning
        self.last_p_fail         = p_fail
        self._code_interventions += 1

        return (
            f"{result}\n\n"
            f"⚠️ BRAIN:\n{warning}\n\n"
            f"Fix this issue and call python_exec again with the corrected implementation."
        )

    def on_chunk(self, text: str) -> str | None:
        """Hook for text-only tasks: update p_fail for display, never fire.

        Trajectory resemblance alone is not concrete evidence — on_chunk
        cannot name a specific mistake so it must not fire an intervention.
        Use before_tool_call + on_tool_call for concrete evidence instead.
        """
        self._pending_trace += text
        self._current_turn  += 1

        if self._current_turn % 5 != 0:
            return None

        p_fail, _, _, _ = self._traj_store.predict_with_context(self._pending_trace)
        if p_fail is not None:
            self.last_p_fail = p_fail

        return None

    def _run_probe(self, code: str) -> list[str]:
        if not self._task_probe_fn:
            return []
        try:
            ns: dict = {}
            with _suppress_stdout():
                exec(compile(code, "<brain_probe>", "exec"), ns)  # noqa: S102
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
            parts.append("Fix this issue and call python_exec again with corrected code.")
        elif knn_warning:
            for line in knn_warning.splitlines():
                line = line.strip()
                if line and not line.startswith("[BRAIN") and not line.startswith("Avoid"):
                    parts.append(f"Warning (similar past failure): {line[:160]}")
                    break
        return "\n".join(parts) if parts else (
            "This reasoning pattern has caused failures before — double-check correctness."
        )

    # ── wrap API ──────────────────────────────────────────────────────────────

    def wrap(self, agent_fn: Callable, verifier: Callable | None = None) -> Callable:
        """Wrap any agent callable so the brain monitors every call automatically."""
        task_counter = [0]

        def wrapped(prompt: str, **kwargs):
            idx = task_counter[0]
            self.set_task(idx, task=prompt[:300])
            self._current_task_str = prompt[:300]
            self.reset()

            if hasattr(agent_fn, 'monitor'):
                agent_fn.monitor = self

            result = agent_fn(prompt, **kwargs)
            trace  = result[0] if isinstance(result, tuple) else result
            tokens = result[1] if isinstance(result, tuple) else 0

            self.on_chunk(trace)

            if verifier is not None:
                label = 1 if verifier(prompt, trace) >= 0.5 else 0
                self.store(trace, label)

            task_counter[0] += 1
            return result

        wrapped._brain      = self
        wrapped._task_count = task_counter
        return wrapped

    # ── storage ───────────────────────────────────────────────────────────────

    def store(self, trace: str, label: int, metadata: str = "") -> None:
        """Store a completed run. Stores the trace as a LatentTrajectory and
        the current task prompt in the task store for pre-gen comparison."""
        self._traj_store.add(trace, label, metadata)
        self._pending_trace = trace

        # Store task prompt for task-level similarity matching
        if self._current_task_str:
            self._task_store.add(self._current_task_str, label, metadata)

        if label == 0 and metadata:
            # Index the error message so future exec-error matching is on mistake type
            vec = self._embedder([metadata[:500]])[0]
            with self._error_lock:
                self._error_patterns.append((np.asarray(vec, dtype='float32'), metadata[:300]))

            task_str = self._current_task_str
            reason   = metadata  # failure reason / verifier detail

            def _extract():
                self._motif_store.add_learned(trace, task=task_str, reason=reason)

            t = threading.Thread(target=_extract, daemon=True, name="brain-extract")
            t.start()
            # Wait up to 5s — structured motif extraction includes an LLM call.
            t.join(timeout=5.0)

    def store_code(self, code: str, label: int, metadata: str = "") -> None:
        if code.strip():
            self._code_store.add(code[:2000], label, metadata)

    def store_passing_code(self, label: str, code: str) -> None:
        self._passing_codes.append((label, code))
        if code.strip():
            self._code_store.add(code[:2000], 1, metadata=f"passing: {label}")

    def store_result(self, label: int) -> None:
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
        """Receive a chunk of agent reasoning text (called by tool_agent for every
        text block the model emits between tool calls).

        Updates both the BackgroundMonitor buffer (for should_bail / pulse) and
        the live reasoning trajectory (_live_steps), which before_tool_call uses
        to predict failure from the direction of reasoning — not just code patterns.
        """
        with self._lock:
            self._buffer += text

        # Build reasoning trajectory from the actual text the agent writes.
        # Each chunk becomes a LatentStep; drift = change in semantic direction.
        if text and len(text.strip()) >= 40:
            self._pending_trace += text
            vec  = np.asarray(self._embedder([text[:1500]])[0], dtype='float32')
            norm = np.linalg.norm(vec)
            if norm > 1e-9:
                vec = vec / norm
            drift = 0.0
            if self._live_steps:
                drift = max(0.0, min(2.0, 1.0 - float(np.dot(vec, self._live_steps[-1].vec))))
            self._live_steps.append(LatentStep(
                text=text[:300], vec=vec, drift=drift,
                step_type=_classify_step(text), index=len(self._live_steps),
            ))

    @property
    def should_bail(self) -> bool:
        return self._bail_ev.is_set()

    def reset(self) -> None:
        """Reset all per-task state. Call at the start of each new task.

        CRITICAL: clears _trajectory so that get_trajectory() / any(pt.fired ...)
        reflects ONLY the current task, not all prior tasks. Without this, every
        task after the first legitimate fire would appear to have fired.
        """
        with self._lock:
            self._buffer = ""
        with self._traj_lock:
            self._trajectory = []        # fire detection is per-task, not global
        self._bail_ev.clear()
        self.last_p_fail         = None
        self.last_warning        = None
        self.last_fire           = None   # clear fire audit record
        self._pending_trace      = ""
        self._last_code_warning  = ""
        self._code_interventions = 0
        self._current_turn       = 0
        self._turn_count         = 0
        self._fire_turn          = None
        self._stall_streak       = 0
        self._stall_fired        = False
        self._live_steps         = []

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
        """Immediate trajectory check: update p_fail for display only.

        Trajectory resemblance is NEVER a fire trigger. This method logs
        last_p_fail for the display layer but never sets the bail flag or
        generates a warning from trajectory alone.
        """
        with self._lock:
            buf = self._buffer
        if len(buf) < self._min_chars:
            return False
        if len(self._traj_store._runs) < 5:
            return False
        p_fail, _, _, _ = self._traj_store.predict_with_context(buf)
        if p_fail is not None:
            self.last_p_fail = p_fail
        # Never fire — trajectory alone cannot trigger bail.
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
        """Background thread: periodically updates last_p_fail for display.

        Trajectory resemblance is NEVER a fire trigger. The loop computes
        p_fail purely for logging — it never sets the bail flag or generates
        intervention messages from trajectory alone.
        """
        while not self._stop_ev.is_set():
            time.sleep(self._check_interval)
            with self._lock:
                buf = self._buffer
            if len(buf) < self._min_chars:
                continue
            p_fail, _, _, _ = self._traj_store.predict_with_context(buf)
            if p_fail is not None:
                self.last_p_fail = p_fail
            # Never fire on trajectory: p_fail is context only.

    # ── wrap_any API ──────────────────────────────────────────────────────────

    def wrap_any(self, agent_or_model, retry_fn=None, cot: bool = False):
        """Wrap any model string or callable with brain monitoring."""
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

    def seed(self, items: list[dict]) -> None:
        """Pre-populate trajectory and task stores with known failure/pass patterns.

        For abstract failure patterns (regexes, named motifs), use seed_motifs() instead.
        """
        for item in items:
            trace = item["trace"]
            label = item["label"]
            meta  = item.get("metadata", "")
            self._traj_store.add(trace, label, meta)
            self._task_store.add(trace, label, meta)
            if label == 0 and meta:
                vec = np.asarray(self._embedder([meta[:500]])[0], dtype='float32')
                with self._error_lock:
                    self._error_patterns.append(
                        (np.asarray(vec, dtype='float32'), meta[:300])
                    )

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
            gen_prompt   = _inject_cot(prompt) if cot else prompt
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
        """Extended thinking — model's internal CoT streams to the brain."""
        import anthropic
        from .agents import _anthropic_call
        brain  = self
        _retry = retry_fn

        def monitored(prompt: str) -> tuple:
            brain.reset()
            from trace_use import agents as _ag
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
                warn = brain.last_warning or ""
                r_fn = _retry or (lambda p: _anthropic_call(model, p, max_tokens=2048))
                rp   = brain._RETRY_PREFIX.format(warning=warn) + prompt if warn else prompt
                return r_fn(rp)
            return full_trace, total_tokens

        monitored.__name__ = f"brain_thinking({model})"
        monitored._brain   = brain
        return monitored


# ── utility ───────────────────────────────────────────────────────────────────

class _suppress_stdout:
    def __enter__(self):
        import contextlib
        self._cm = contextlib.redirect_stdout(io.StringIO())
        self._cm.__enter__()
    def __exit__(self, *a):
        self._cm.__exit__(*a)
