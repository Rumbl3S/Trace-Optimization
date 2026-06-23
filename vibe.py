"""vibe.py — Brain-monitored vibecoding session.

Wraps claude-sonnet-4-6 (extended thinking) with a FailureStore so that when
Claude starts reasoning down a path it has already tried and failed, the brain
fires and injects exactly what went wrong last time.

Usage::

    from vibe import VibeSession

    def verify(code: str) -> tuple[bool, str]:
        # run your tests, return (passed, feedback)
        ...

    session = VibeSession("build a rate limiter", verify)
    session.attempt("implement it")
    session.attempt("tests still failing — the window resets wrong")
    session.summary()
"""
from __future__ import annotations

import re
import subprocess
import sys
import textwrap
import time
from dataclasses import dataclass, field
from typing import Callable

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

from brain import BrainAgent
from agents import _build_openai

console = Console()

_SYSTEM = """You are an expert Python developer in a vibecoding session.
Think carefully through your approach before writing code.
Always write the complete, runnable implementation — no stubs, no TODOs.
Put your final code in a single ```python ... ``` block.
"""


@dataclass
class Attempt:
    number:      int
    prompt:      str
    thinking:    str
    response:    str
    code:        str | None
    passed:      bool
    feedback:    str
    p_fail:      float | None
    brain_fired: bool
    duration_s:  float


class VibeSession:
    """A coding session where the brain remembers every failure.

    Every attempt's thinking trace is embedded and stored. When Claude starts
    reasoning toward an approach it has already failed with, the brain fires
    mid-generation, stops the attempt, and retries with explicit context about
    what went wrong before.
    """

    def __init__(
        self,
        goal: str,
        verifier: Callable[[str], tuple[bool, str]],
        model: str = "claude-sonnet-4-6",
        thinking_budget: int = 8000,
        use_cot: bool = False,
        threshold: float = 0.38,
        k: int = 3,
    ):
        self.goal     = goal
        self.verifier = verifier
        self.model    = model
        self._budget  = thinking_budget
        self._use_cot = use_cot
        self.history: list[Attempt] = []

        _THINKING = {"claude-sonnet-4-6", "claude-opus-4-8", "claude-opus-4-7"}

        embedder   = _build_openai()
        self.brain = BrainAgent(
            embedder,
            k=k,
            threshold=threshold,
            check_interval=2.0,
            min_chars=200,
        )
        base = "-".join(model.split("-")[:4])
        if base in _THINKING or any(m in model for m in _THINKING):
            self._agent  = self.brain.wrap_thinking(model)
            self._mode   = "thinking"
        else:
            self._agent  = self.brain.wrap_any(model, cot=use_cot)
            self._mode   = "cot" if use_cot else "stream"

        mode_label = {"thinking": "extended thinking", "cot": "CoT stream", "stream": "stream"}[self._mode]
        console.print(Panel(
            f"[bold cyan]VibeSession[/]\n[dim]{goal[:120]}[/]\n\n"
            f"Model: {model} ({mode_label})  |  Brain threshold: {threshold}  |  k={k}",
            box=box.ROUNDED,
        ))

    # ── public API ────────────────────────────────────────────────────────────

    def attempt(self, user_message: str) -> Attempt:
        """Run one coding attempt. Returns the Attempt record."""
        n = len(self.history) + 1
        console.rule(f"[bold]Attempt {n}[/]")
        console.print(f"[dim]{user_message[:120]}[/]")

        # Build prompt: include goal + failure memory + current message
        prompt = self._build_prompt(user_message, n)

        t0 = time.time()
        result = self._agent(prompt)
        duration = time.time() - t0

        thinking, response = self._split_trace(result)
        code     = _extract_code(response)
        p_fail   = self.brain.last_p_fail
        fired    = self.brain.should_bail

        # Verify
        if code:
            passed, feedback = self.verifier(code)
        else:
            passed, feedback = False, "No code block found in response."

        label = int(passed)
        trace = f"[THINKING]\n{thinking}\n[RESPONSE]\n{response}"
        self.brain.store(trace, label)

        status = "[green]PASS[/]" if passed else "[red]FAIL[/]"
        fire   = " [yellow]⚡ BRAIN FIRED[/]" if fired else ""
        pfail  = f"  p_fail={p_fail:.2f}" if p_fail is not None else ""
        console.print(f"{status}{fire}{pfail}  ({duration:.1f}s)")

        if not passed:
            console.print(Panel(
                feedback[:600],
                title="[red]Test output[/]",
                border_style="red",
                box=box.SIMPLE,
            ))

        attempt = Attempt(
            number=n,
            prompt=user_message,
            thinking=thinking,
            response=response,
            code=code,
            passed=passed,
            feedback=feedback,
            p_fail=p_fail,
            brain_fired=fired,
            duration_s=duration,
        )
        self.history.append(attempt)
        return attempt

    def summary(self) -> None:
        """Print session summary table."""
        t = Table(title="Session Summary", box=box.SIMPLE_HEAVY)
        t.add_column("#",        style="dim",    width=3)
        t.add_column("Result",   width=6)
        t.add_column("p_fail",   width=7)
        t.add_column("Brain",    width=6)
        t.add_column("Time",     width=6)
        t.add_column("Feedback", no_wrap=False, max_width=60)

        for a in self.history:
            res   = "[green]PASS[/]" if a.passed else "[red]FAIL[/]"
            pf    = f"{a.p_fail:.2f}" if a.p_fail is not None else "—"
            fire  = "[yellow]⚡[/]" if a.brain_fired else "—"
            fb    = a.feedback[:80].replace("\n", " ") if not a.passed else "✓"
            t.add_row(str(a.number), res, pf, fire, f"{a.duration_s:.1f}s", fb)

        console.print(t)
        passes = sum(1 for a in self.history if a.passed)
        fires  = sum(1 for a in self.history if a.brain_fired)
        console.print(
            f"\n[bold]{passes}/{len(self.history)} passed[/]  "
            f"[yellow]{fires} brain fires[/]  "
            f"[dim]{self.brain.n_stored} traces stored "
            f"({self.brain.n_pass}✓ {self.brain.n_fail}✗)[/]"
        )

    @property
    def last_code(self) -> str | None:
        for a in reversed(self.history):
            if a.code and a.passed:
                return a.code
        for a in reversed(self.history):
            if a.code:
                return a.code
        return None

    # ── internals ────────────────────────────────────────────────────────────

    def _build_prompt(self, user_message: str, attempt_n: int) -> str:
        parts = [_SYSTEM, f"\n## Goal\n{self.goal}\n"]

        # Inject failure history so the model knows what not to repeat
        fails = [a for a in self.history if not a.passed]
        if fails:
            parts.append("\n## Previous failed attempts — do NOT repeat these approaches\n")
            for a in fails[-3:]:  # last 3 failures
                parts.append(
                    f"\n### Attempt {a.number} — FAILED\n"
                    f"**Approach summary** (from thinking):\n"
                    f"{_first_lines(a.thinking, 8)}\n\n"
                    f"**Test failure:**\n{a.feedback[:400]}\n"
                )

        parts.append(f"\n## Current request\n{user_message}")
        return "\n".join(parts)

    def _split_trace(self, result) -> tuple[str, str]:
        if isinstance(result, tuple):
            full = result[0]
        else:
            full = str(result)
        # _wrap_thinking returns "[THINKING]\n...\n[RESPONSE]\n..."
        if "[RESPONSE]" in full:
            parts    = full.split("[RESPONSE]", 1)
            thinking = parts[0].replace("[THINKING]", "").strip()
            response = parts[1].strip()
        else:
            # streaming / CoT agent returns raw text as the trace
            thinking = ""
            response = full
        return thinking, response


# ── helpers ───────────────────────────────────────────────────────────────────

def _extract_code(text: str) -> str | None:
    blocks = re.findall(r"```(?:python|py)?\n(.*?)```", text, re.DOTALL)
    return blocks[-1].strip() if blocks else None


def _first_lines(text: str, n: int) -> str:
    lines = text.strip().splitlines()
    excerpt = "\n".join(lines[:n])
    return textwrap.indent(excerpt, "  ")


def run_tests(code: str, test_code: str) -> tuple[bool, str]:
    """Write code + tests to a temp file and run pytest. Returns (passed, output)."""
    import tempfile, os
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py",
                                     delete=False, dir="/tmp") as f:
        f.write(code + "\n\n" + test_code)
        path = f.name
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", path, "-x", "--tb=short", "-q"],
            capture_output=True, text=True, timeout=15,
        )
        output = (proc.stdout + proc.stderr).strip()
        passed = proc.returncode == 0
        return passed, output
    except subprocess.TimeoutExpired:
        return False, "Tests timed out (>15s)"
    finally:
        os.unlink(path)
