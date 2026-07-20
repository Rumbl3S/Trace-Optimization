"""TrajectoryDetector — pre-task motif relevance check with pre-prompt injection.

Before the LLM writes any code, compare the developer's task description against
stored failure motifs. If a motif's required_condition is concretely found in the
task text, inject a KNOWN PITFALLS block into the prompt.

This operates at task level (before generation starts), complementing BrainAgent
which operates at tool-call level (before execution):

  TrajectoryDetector.inject(task)     — "does this task description suggest the gap?"
  BrainAgent.before_tool_call(...)    — "does this proposed code exhibit the gap?"

Both signals are independent and can fire on the same task.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .brain import FailureMotif, _STOPWORDS, _parse_json_first
from .config import DetectorConfig


_TRAJECTORY_JUDGE_PROMPT = """\
A past failure revealed this logical pattern:

  Pattern name:  {motif_name}
  Logical issue: {motif_description}
  Applies when task explicitly mentions: "{required_condition}"
  Recommendation: {recommendation}

New developer task:
  "{task}"

Question: Does this task's description explicitly mention the same logical requirement
that caused the past failure?

Steps:
1. Search the task text for exact wording related to "{required_condition}".
   Quote it word-for-word from the task text above.
2. If you find concrete matching text → relevant=true, fill in evidence_quote and warning.
3. If the task does not explicitly mention this requirement → relevant=false.

Important: Do NOT answer relevant=true because the domain is similar or the task might
implicitly need it. Only quote text that is explicitly present in the task description.

Return JSON only (no markdown, no extra text):
{{
  "relevant": <true|false>,
  "confidence": <0.0-1.0>,
  "evidence_quote": "<exact phrase from the task description that matches, or empty string>",
  "warning": "<one concrete sentence about what to watch out for in this specific task, or empty string>"
}}
"""

_VAGUE_PHRASES = frozenset({
    "task implies", "likely", "may need", "might", "possibly",
    "perhaps", "similar to", "probably", "should consider",
    "could fail", "seems like", "appears to", "suggests that",
    "would need", "general requirement",
})


@dataclass
class MotifMatch:
    """A motif that is concretely relevant to a given task description."""
    motif:          FailureMotif
    confidence:     float
    evidence_quote: str          # exact phrase from the task that triggered this motif
    warning:        str          # one-sentence warning tailored to the current task


def _word_overlap(quote: str, target: str) -> float:
    q_words = set(re.findall(r"\w+", quote.lower())) - _STOPWORDS
    if not q_words:
        return 0.0
    t_words = set(re.findall(r"\w+", target.lower()))
    return len(q_words & t_words) / len(q_words)


def _validate_match(result: dict, task: str, threshold: float) -> bool:
    """Return True only when the judge produced concrete, grounded evidence."""
    if not result.get("relevant"):
        return False
    if float(result.get("confidence", 0)) < threshold:
        return False
    quote   = (result.get("evidence_quote") or "").strip()
    warning = (result.get("warning")        or "").strip()
    if not quote or len(warning) < 10:
        return False
    # Evidence must be grounded in the actual task text
    grounded = (
        quote.lower() in task.lower()
        or _word_overlap(quote, task) >= 0.60
    )
    if not grounded:
        return False
    # Reject vague speculative language
    evidence_lower = (quote + " " + warning).lower()
    if any(p in evidence_lower for p in _VAGUE_PHRASES):
        return False
    return True


class TrajectoryDetector:
    """Pre-task logical similarity check against the learned motif store.

    Usage::

        detector = TrajectoryDetector(motif_store, embedder)

        # Enriches the prompt with a KNOWN PITFALLS block if relevant motifs exist.
        enriched_prompt, matches = detector.inject(task_prompt)

        # Inspect what would fire without modifying the prompt.
        matches = detector.check(task_prompt)
    """

    def __init__(
        self,
        motif_store,                         # MotifStore or PersistentMotifStore
        embedder:    Callable,
        config:      DetectorConfig | None = None,
    ):
        self._store = motif_store
        self._embed = embedder
        self._cfg   = config or DetectorConfig()

    # ── public API ────────────────────────────────────────────────────────────

    def check(self, task: str) -> list[MotifMatch]:
        """Return all motifs that are logically relevant to this task description.

        Makes one cheap LLM call per candidate motif (up to retrieval_top_k).
        Returns an empty list when no relevant motifs exist or the store is empty.
        """
        candidates = self._store.retrieve(
            task=task, reasoning="", code="",
            top_k=self._cfg.retrieval_top_k,
            min_sim=self._cfg.retrieval_min_sim,
        )
        matches = []
        for motif in candidates:
            result = self._call_judge(motif, task)
            if result and _validate_match(result, task, self._cfg.judge_threshold):
                matches.append(MotifMatch(
                    motif          = motif,
                    confidence     = float(result.get("confidence", 0)),
                    evidence_quote = (result.get("evidence_quote") or "").strip(),
                    warning        = (result.get("warning")        or "").strip(),
                ))
        return matches

    def inject(self, task: str) -> tuple[str, list[MotifMatch]]:
        """Return (enriched_prompt, matches).

        If relevant motifs are found, prepends a KNOWN PITFALLS block to the prompt.
        When no relevant motifs exist, returns the original prompt unchanged.
        """
        matches = self.check(task)
        if not matches:
            return task, []
        return f"{_format_pitfalls(matches)}\n\n{task}", matches

    # ── internals ─────────────────────────────────────────────────────────────

    def _call_judge(self, motif: FailureMotif, task: str) -> dict | None:
        try:
            from .agents import _llm_call
        except ImportError:
            return None
        prompt = _TRAJECTORY_JUDGE_PROMPT.format(
            motif_name         = motif.name,
            motif_description  = motif.description,
            required_condition = motif.required_condition or motif.description,
            recommendation     = motif.recommendation,
            task               = task[:600],
        )
        try:
            text, _ = _llm_call(
                self._cfg.provider, self._cfg.model, prompt,
                max_tokens=self._cfg.max_tokens,
            )
            text = text.strip()
            if text.startswith("```"):
                text = re.sub(r"^```[a-z]*\n?", "", text)
                text = re.sub(r"\n?```$", "", text)
            return _parse_json_first(text)
        except Exception as e:
            print(f"[TRAJECTORY JUDGE ERROR] motif={motif.id!r} err={e!r}")
            return None


# ── Formatting (standalone so demo_session.py can import it directly) ─────────

def _format_pitfalls(matches: list[MotifMatch]) -> str:
    lines = ["⚠️  KNOWN PITFALLS — based on recorded failures:", ""]
    for i, m in enumerate(matches, 1):
        lines += [
            f"  [{i}] {m.motif.name}",
            f"      Why this applies: \"{m.evidence_quote}\"",
            f"      Watch out for:    {m.warning}",
            f"      Recommendation:   {m.motif.recommendation}",
            "",
        ]
    lines.append("Address these before writing your implementation.")
    return "\n".join(lines)
