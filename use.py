"""use.py — One-import brain wrapper for any day-to-day model use.

  from use import BrainSession

  session = BrainSession()                    # reads ANTHROPIC_API_KEY, OPENAI_API_KEY
  agent   = session.agent("claude-haiku-4-5-20251001")   # or any model / callable

  # Your normal loop:
  result = agent("write X")                   # predict → generate → return
  print(f"p_fail={session.p_fail:.0%}")       # how likely was this to fail?

  success = my_tests_pass(result)
  session.teach(success)                      # one line — brain learns the outcome

If p_fail exceeds the threshold the brain automatically prepends failure context
to the prompt so the model starts smarter, not you having to remember why it
failed last time.

Cost: one OpenAI embedding call (~$0.0001) per task. No extra LLM calls.
Works with: any Anthropic model string, any OpenAI model string, any callable.
"""
from __future__ import annotations

import os
import re
from typing import Callable

from brain import BrainAgent, FailureStore
from agents import _build_openai


# ── Routing ──────────────────────────────────────────────────────────────────

_DEFAULT_SYSTEM = (
    "You are a helpful assistant. "
    "When writing code, output it in a single ```python ... ``` code block. "
    "Embed your reasoning as comments inside the block."
)


def _make_agent(model_or_fn, system: str = _DEFAULT_SYSTEM) -> Callable:
    """Turn a model name or callable into a (prompt) -> (text, tokens) callable."""
    if callable(model_or_fn) and not isinstance(model_or_fn, str):
        fn = model_or_fn
        def _wrap(prompt: str) -> tuple[str, int]:
            r = fn(prompt)
            if isinstance(r, tuple):
                return r[0], r[1] if len(r) > 1 else 0
            return str(r), 0
        return _wrap

    model: str = model_or_fn
    if "claude" in model.lower():
        import anthropic
        client = anthropic.Anthropic()
        needs_thinking = any(m in model for m in ("sonnet-4", "opus-4"))
        def _claude(prompt: str) -> tuple[str, int]:
            kwargs: dict = {}
            if needs_thinking:
                kwargs["thinking"] = {"type": "enabled", "budget_tokens": 5000}
            msg = client.messages.create(
                model=model, max_tokens=4096,
                system=system,
                messages=[{"role": "user", "content": prompt}],
                **kwargs,
            )
            text = "".join(
                b.text if hasattr(b, "text") else getattr(b, "thinking", "")
                for b in msg.content
            )
            return text, msg.usage.input_tokens + msg.usage.output_tokens
        return _claude

    if "gpt" in model.lower() or "o1" in model.lower() or "o3" in model.lower():
        from openai import OpenAI
        client = OpenAI()
        def _openai(prompt: str) -> tuple[str, int]:
            rsp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
            )
            text   = rsp.choices[0].message.content or ""
            tokens = rsp.usage.total_tokens if rsp.usage else 0
            return text, tokens
        return _openai

    raise ValueError(f"Unknown model: {model!r}. Pass a callable for custom models.")


# ── BrainSession ─────────────────────────────────────────────────────────────

class BrainSession:
    """Brain wrapper for any model. Predicts failures from past traces.

    Usage::

        session = BrainSession()
        agent   = session.agent("claude-haiku-4-5-20251001")

        result  = agent("write a parser for X")
        print(f"p_fail={session.p_fail:.0%}  fired={session.fired}")

        success = run_tests(result)
        session.teach(success)

    Args:
        threshold:  p_fail above this → inject failure context into prompt.
                    Default 0.35. Lower = more aggressive intervention.
        k:          kNN neighbours to consider. Default 5.
        auto_teach: If True and the agent returns a (text, label) tuple,
                    store the label automatically without calling teach().
    """

    def __init__(
        self,
        threshold: float = 0.35,
        k: int = 5,
        auto_teach: bool = False,
    ):
        self._embedder    = _build_openai()
        self._brain       = BrainAgent(
            self._embedder, k=k, threshold=threshold,
            check_interval=9999, min_chars=999999,   # pre-gen only; no mid-gen thread
        )
        self._auto_teach  = auto_teach
        self._last_text   = ""
        self._last_prompt = ""

    # ── public ───────────────────────────────────────────────────────────────

    def agent(self, model_or_fn, system: str = _DEFAULT_SYSTEM) -> Callable[[str], str]:
        """Return a drop-in replacement for your model that adds brain monitoring.

        The returned callable takes a prompt string and returns the response text.
        Before generating it queries the failure store; if p_fail is high it
        injects what went wrong last time directly into the prompt.
        """
        _inner = _make_agent(model_or_fn, system=system)
        session = self

        def _monitored(prompt: str) -> str:
            session._last_prompt = prompt

            # Pre-generation: predict
            p_fail, warning = session._brain.failure_store.query(prompt)
            session._p_fail  = p_fail
            session._warning = warning
            session._fired   = warning is not None

            # Inject context if brain flagged high risk
            if warning:
                augmented = (
                    f"[BRAIN WARNING — P(fail)={p_fail:.0%}]\n"
                    f"A similar task previously failed:\n{warning}\n\n"
                    f"Original request:\n{prompt}"
                )
            else:
                augmented = prompt

            # Generate
            result = _inner(augmented)
            text   = result[0] if isinstance(result, tuple) else str(result)
            session._last_text = text
            return text

        _monitored.__name__ = f"brain({model_or_fn if isinstance(model_or_fn, str) else model_or_fn.__name__})"
        return _monitored

    def teach(self, success: bool | int, metadata: str = "") -> None:
        """Tell the brain whether the last generation succeeded.

        metadata: optional diagnosis of what went wrong (or right). Shown
        verbatim in future warnings when this trace matches a new task.
        """
        if not self._last_text:
            return
        self._brain.store(self._last_text, int(bool(success)), metadata)
        self._last_text = ""

    def seed(self, items: list[dict]) -> None:
        """Pre-populate with known failure/pass patterns before running any tasks.

        items: list of {'trace': str, 'label': int, 'metadata': str}
        """
        self._brain.seed(items)

    # ── inspection ───────────────────────────────────────────────────────────

    @property
    def p_fail(self) -> float | None:
        """P(fail) computed before the last generation. None = not enough data."""
        return getattr(self, "_p_fail", None)

    @property
    def fired(self) -> bool:
        """True if the brain injected failure context into the last prompt."""
        return getattr(self, "_fired", False)

    @property
    def warning(self) -> str | None:
        """The warning text injected (if any)."""
        return getattr(self, "_warning", None)

    @property
    def n_stored(self) -> int:
        return self._brain.n_stored

    @property
    def n_pass(self) -> int:
        return self._brain.n_pass

    @property
    def n_fail(self) -> int:
        return self._brain.n_fail

    def summary(self) -> None:
        """Print a one-line summary of the session so far."""
        store = self._brain.failure_store
        _, labels = store.all_vecs()
        n_pass = sum(l == 1 for l in labels)
        n_fail = sum(l == 0 for l in labels)
        print(
            f"BrainSession: {len(labels)} tasks stored  "
            f"{n_pass}✓ {n_fail}✗  "
            f"threshold={self._brain._store._threshold}"
        )

    def store_snapshot(self) -> tuple:
        """Return (embeddings, labels) for plotting or analysis."""
        return self._brain.failure_store.all_vecs()
