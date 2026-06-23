"""brain.py — Brain Agent: inference-time failure detection for any LLM agent.

ARCHITECTURE
────────────
One store. One embedder. No extra API calls.

The brain embeds every stored trace (pass or fail) using the same OpenAI
embedder you already have. While a new agent is generating, the brain embeds
the partial trace every N seconds and asks: does this look like a past failure?

If the partial trace is kNN-similar to stored failures (p_fail >= threshold),
the brain stops generation mid-stream and retries immediately, injecting a
warning that shows exactly which past failure it resembles and how it ended.

After verification, brain.store(trace, label) adds the outcome to the store.
Future tasks with similar trajectories get caught before the wrong answer.

IMPORTABLE API
──────────────
    from brain import BrainAgent
    from agents import _build_openai

    brain = BrainAgent(_build_openai())
    agent = brain.wrap_any("claude-haiku-4-5-20251001")   # any model
    # OR
    agent = brain.wrap_any(haiku)                          # any callable

    result = agent(prompt)         # brain monitors, retries if needed
    brain.store(trace, label)      # teach the outcome
"""
from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Callable

import numpy as np


# ── Failure store ─────────────────────────────────────────────────────────────

@dataclass
class StoredTrace:
    trace_id: str
    label:    int          # 1 = pass, 0 = fail
    excerpt:  str          # first 300 chars — shown in warning
    tail:     str          # last 300 chars — what the wrong answer looked like
    vec:      np.ndarray = field(repr=False)


class FailureStore:
    """kNN store over trace embeddings.

    Stores every verified trace with its pass/fail label. Queries the partial
    trace as it streams in. When the partial trace is closer to past failures
    than past passes, fires a bail.

    One embedder call per check interval (background thread). No extra models.
    """

    def __init__(self, embedder: Callable, k: int = 5, threshold: float = 0.45):
        self._embedder  = embedder
        self._k         = k
        self._threshold = threshold
        self._traces:   list[StoredTrace]  = []
        self._matrix:   np.ndarray | None  = None
        self._lock      = threading.Lock()

    def add(self, trace: str, label: int) -> None:
        vec = self._embedder([trace[-1500:]])[0]
        entry = StoredTrace(
            trace_id=uuid.uuid4().hex[:8],
            label=label,
            excerpt=trace[:300],
            tail=trace[-300:],
            vec=vec,
        )
        with self._lock:
            self._traces.append(entry)
            self._matrix = np.stack([t.vec for t in self._traces])

    def query(self, partial: str) -> tuple[float | None, str | None]:
        """Embed partial trace, find kNN, return (p_fail, warning_or_None)."""
        with self._lock:
            if self._matrix is None:
                return None, None
            labels = [t.label for t in self._traces]
            if len(set(labels)) < 2:
                return None, None   # need both pass and fail before we can discriminate
            mat    = self._matrix.copy()
            traces = list(self._traces)

        vec  = self._embedder([partial[-1500:]])[0]
        sims = mat @ vec
        k    = min(self._k, len(traces))
        top  = np.argpartition(sims, -k)[-k:]
        nbrs = [traces[i] for i in top]

        n_fail = sum(1 for t in nbrs if t.label == 0)
        p_fail = n_fail / k

        if p_fail < self._threshold:
            return p_fail, None

        # Find the most similar failed trace to show as context
        fail_nbrs = sorted(
            [traces[i] for i in top if traces[i].label == 0],
            key=lambda t: float(sims[traces.index(t)]),
            reverse=True,
        )
        best = fail_nbrs[0] if fail_nbrs else None
        warning = _build_warning(p_fail, best) if best else None
        return p_fail, warning

    @property
    def n(self) -> int:
        return len(self._traces)

    @property
    def n_fail(self) -> int:
        return sum(1 for t in self._traces if t.label == 0)

    @property
    def n_pass(self) -> int:
        return sum(1 for t in self._traces if t.label == 1)

    def all_vecs(self) -> tuple[np.ndarray | None, list[int]]:
        """Return (matrix, labels) snapshot for visualization."""
        with self._lock:
            if not self._traces:
                return None, []
            return self._matrix.copy(), [t.label for t in self._traces]


def _build_warning(p_fail: float, failed: StoredTrace) -> str:
    return (
        f"[BRAIN] P(fail) = {p_fail:.0%} — this trajectory resembles a past failure:\n"
        f"  \"{failed.excerpt[:200]}\"\n"
        f"  Past wrong answer: \"{failed.tail[-150:]}\"\n"
        f"  ↳ Reconsider your approach before continuing."
    )


# ── BrainAgent ────────────────────────────────────────────────────────────────

class BrainAgent:
    """Inference-time failure detector. Wraps any agent or model.

    Brain embeds the partial trace every check_interval seconds.
    When p_fail >= threshold: stops generation, retries with failure context.
    After verification: brain.store(trace, label) teaches the outcome.

    API::
        brain = BrainAgent(embedder)
        agent = brain.wrap_any("claude-haiku-4-5-20251001")
        agent = brain.wrap_any(haiku)

        result = agent(prompt)
        brain.store(trace, label)
    """

    _RETRY_PREFIX = (
        "Your previous reasoning was going in the wrong direction. "
        "The brain detected it resembled a past failure:\n\n"
        "{warning}\n\n"
        "Start fresh with a different approach. "
        "Write the complete implementation in a ```python code block.\n\n"
    )

    def __init__(self, embedder: Callable, k: int = 5, threshold: float = 0.45,
                 check_interval: float = 1.0, min_chars: int = 200):
        self._store          = FailureStore(embedder, k=k, threshold=threshold)
        self._embedder       = embedder
        self._check_interval = check_interval
        self._min_chars      = min_chars

        self._buffer   = ""
        self._lock     = threading.Lock()
        self._bail_ev  = threading.Event()
        self._stop_ev  = threading.Event()
        self._thread:  threading.Thread | None = None

        self.last_p_fail:  float | None = None
        self.last_warning: str   | None = None
        self._pending_trace: str        = ""

    # ── universal entry point ─────────────────────────────────────────────────

    def wrap_any(self, agent_or_model, retry_fn=None, cot: bool = False):
        """Wrap any model string or callable.

        Model routing:
          'claude-sonnet-4-6' / 'claude-opus-4-8' → native extended thinking (no CoT needed)
          Any other string                          → streaming
          Any callable                              → streaming

        Args:
            cot: Inject a step-by-step CoT prompt automatically. Improves trace
                 richness for embedding but also improves accuracy — disable for
                 demos where you need realistic failure rates from weaker models.
        """
        _THINKING = {"claude-sonnet-4-6", "claude-opus-4-8", "claude-opus-4-7"}

        if isinstance(agent_or_model, str):
            model = agent_or_model
            base  = "-".join(model.split("-")[:4])
            if base in _THINKING or any(m in model for m in _THINKING):
                return self._wrap_thinking(model, retry_fn)
            else:
                return self._wrap_model(model, retry_fn, cot=cot)
        else:
            return self._wrap_callable(agent_or_model, retry_fn, cot=cot)

    def wrap_thinking(self, model: str = "claude-sonnet-4-6",
                      budget: int = 5000, retry_fn=None):
        """Wrap an Anthropic extended-thinking model.

        The model's internal chain-of-thought streams directly to the brain.
        No CoT prompt needed — this IS the trace.
        """
        return self._wrap_thinking(model, retry_fn)

    def store(self, trace: str, label: int) -> None:
        """Teach the brain the verified outcome. Call after every task."""
        self._store.add(trace, label)

    # ── BackgroundMonitor-compatible interface ────────────────────────────────

    def push(self, text: str) -> None:
        with self._lock: self._buffer += text

    @property
    def should_bail(self) -> bool:
        return self._bail_ev.is_set()

    def reset(self) -> None:
        with self._lock: self._buffer = ""
        self._bail_ev.clear()
        self.last_p_fail  = None
        self.last_warning = None
        self._pending_trace = ""

    def store_result(self, label: int) -> None:
        if self._pending_trace:
            self.store(self._pending_trace, label)

    def finalize(self, trace: str) -> None:
        self._pending_trace = trace

    def get_intervention(self) -> str | None:
        return self.last_warning

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

    # ── background monitoring loop ────────────────────────────────────────────

    def _loop(self) -> None:
        while not self._stop_ev.is_set():
            time.sleep(self._check_interval)
            with self._lock:
                buf = self._buffer

            if len(buf) < self._min_chars:
                continue

            p_fail, warning = self._store.query(buf)

            if p_fail is not None:
                self.last_p_fail = p_fail
            if warning:
                self.last_warning = warning
                self._bail_ev.set()
                return

    # ── wrap implementations ──────────────────────────────────────────────────

    def _wrap_callable(self, agent_fn, retry_fn=None, cot: bool = False):
        """Wrap any callable agent — tool_agent, streaming_agent, or plain function.

        If the agent has a `.monitor` attribute (tool_agent, streaming_agent),
        it gets set so the agent pushes chunks directly to the brain.
        Otherwise the brain monitors via the timed loop only.
        """
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
        """Anthropic extended thinking — model's internal CoT exposed natively."""
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
                            if delta is None: continue
                            dtype = getattr(delta, 'type', '')
                            if dtype == 'thinking_delta':
                                chunk = getattr(delta, 'thinking', '')
                                thinking_acc += chunk
                                with brain._lock: brain._buffer += chunk
                            elif dtype == 'text_delta':
                                chunk = getattr(delta, 'text', '')
                                text_acc += chunk
                                with brain._lock: brain._buffer += chunk
                        if brain.should_bail:
                            bailed = True; break
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
                full_trace += f"\n[EARLY_EXIT]"
                r_fn = _retry or (lambda p: _ag._anthropic_call(model, p, max_tokens=2048))
                rp   = brain._RETRY_PREFIX.format(warning=warn) + prompt if warn else prompt
                return r_fn(rp)
            return full_trace, total_tokens

        monitored.__name__ = f"brain_thinking({model})"
        monitored._brain   = brain
        return monitored

    # ── accessors ─────────────────────────────────────────────────────────────

    @property
    def failure_store(self) -> FailureStore: return self._store
    @property
    def n_stored(self) -> int:               return self._store.n
    @property
    def n_fail(self) -> int:                 return self._store.n_fail
    @property
    def n_pass(self) -> int:                 return self._store.n_pass


# ── CoT injection (models without native thinking) ────────────────────────────

def _inject_cot(prompt: str) -> str:
    _code_keywords = ("function", "class", "implement", "write a", "def ", "code")
    is_code = any(kw in prompt.lower() for kw in _code_keywords)
    if is_code:
        suffix = (
            "\n\nWork through your approach step by step, explaining your reasoning. "
            "Then provide the complete implementation in a ```python code block."
        )
    else:
        suffix = (
            "\n\nWork through this step by step, showing each reasoning step. "
            "Give your final answer as 'ANSWER: ...'."
        )
    return prompt + suffix
