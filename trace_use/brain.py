"""brain.py — Learned-motif failure detection for tool-use agents.

ARCHITECTURE
────────────
Failure creates a motif. Future reasoning/code is judged against that motif.
If the exact logical gap is concretely present, fire. Otherwise do nothing.

  push(text)             → accumulates recent reasoning trace
  before_tool_call()     → retrieves candidate motifs, runs applicability judge
  on_tool_call()         → reactive stall detection only
  store(trace, label)    → on failure, extracts/updates a reusable motif

The trajectory is the accumulated reasoning text passed to the judge.
It allows detection of logical drift, missing assumptions, and plan-code
contradictions — without kNN scoring, p_fail, or combined weights.

Fire rule: at least one judged motif must produce a concrete, grounded
requirement_quote AND violation_quote. No numeric p_fail trigger.
"""
from __future__ import annotations

import re
import threading
import uuid
from dataclasses import dataclass, field
from typing import Callable

import numpy as np


# ── Prompt templates ──────────────────────────────────────────────────────────

_EXTRACT_MOTIF_PROMPT = """\
A Python implementation just failed. Extract a reusable logical failure concept.

TASK:
{task}

AGENT REASONING (what the agent was thinking before the failure):
{reasoning}

FAILED CODE:
```python
{code}
```

FAILURE REASON:
{reason}

EXISTING KNOWN PATTERNS:
{existing}

---

STEP 1 — Is this the SAME logical mistake as an existing pattern above?
  If YES: set "update_id" to that pattern's id (fill remaining fields for the update).
  If NO:  set "update_id" to null and fill all fields.

The motif must answer three questions:
  1. What must the task explicitly state for this mistake to matter?
  2. What reasoning or code behavior shows the mistake?
  3. What correction should be recommended?

Output ONLY JSON (no markdown, no text after the closing brace):
{{
  "update_id": null,
  "id": "short_snake_case_4_words_max",
  "name": "Human-readable name 5-8 words",
  "description": "One sentence: the logical principle violated — no variable names, no task-specific details",
  "required_condition": "What must the task explicitly state for this to apply? E.g. 'task requires secondary tiebreak when primary sort keys are equal'",
  "violation_condition": "What reasoning or code behavior shows the bug? E.g. 'sort uses single key with no tuple for secondary criterion'",
  "recommendation": "Generalizable one-line fix — no variable names from this specific task",
  "examples": ["brief example task where this applies"],
  "confidence": 0.85
}}

RULES:
- description must generalize: no variable names, field names, or task-specific constants
- required_condition describes what the TASK must say
- violation_condition describes what the CODE or REASONING must show
- recommendation must apply across different domains
- If you cannot identify a concrete logical principle, return {{}}
"""

_APPLICABILITY_JUDGE_PROMPT = """\
A past failure pattern was learned. Decide if this EXACT bug is present in the current run.

TASK:
{task}

AGENT REASONING (latest steps):
{reasoning}

PROPOSED CODE:
```python
{code}
```

PATTERN:
  Name: {motif_name}
  Bug: {motif_description}
  Applies when task says: {required_condition}
  Code/reasoning signature: {violation_condition}
  Fix: {motif_recommendation}

STEPS:
1. Find text in TASK or REASONING that satisfies "{required_condition}". Quote it exactly.
2. Find a line in CODE or REASONING that matches "{violation_condition}". Quote it exactly.
3. If both found → applies=true, fill in the quotes and give a one-line fix.
4. If either is missing → applies=false, leave quotes empty.

Do NOT answer yes because the task domain or vocabulary is similar.
Answer yes only when both quotes are concretely findable in the actual text above.

Return JSON only (no markdown, no extra text after the closing brace):
{{
  "applies": <true|false>,
  "confidence": <0.0-1.0>,
  "requirement_quote": "<exact text from task/reasoning, or empty string>",
  "violation_quote": "<exact code line or reasoning sentence, or empty string>",
  "explanation": "<one sentence why this matches>",
  "recommendation": "<concrete one-line fix, or empty string>"
}}
"""


# ── Stopwords and vague phrase detection ──────────────────────────────────────

_STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "has", "have", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "that", "this", "it", "its", "not", "no",
    "if", "then", "else", "each", "all", "any", "some", "only", "also",
    "as", "than", "when", "where", "which", "who", "how", "what", "i",
})

_VAGUE_JUDGE_PHRASES = frozenset({
    "task implies", "likely", "may need", "could fail", "should consider",
    "similar to", "general requirement", "might", "possibly", "perhaps",
    "seems like", "appears to", "probably", "suggests that", "would need",
})


# ── Deterministic judge validation ────────────────────────────────────────────

def _validate_judge_result(
    result: dict,
    task: str,
    code: str,
    reasoning: str,
    threshold: float = 0.80,
) -> bool:
    """Return True only when the judge produced concrete, grounded proof.

    All of these must hold:
      - applies=True
      - confidence >= threshold (default 0.80)
      - requirement_quote is non-empty and grounded in task/reasoning
      - violation_quote is non-empty and grounded in code/reasoning
      - recommendation is substantive (>= 10 chars)
      - neither quote contains vague speculative language
    """
    if not result.get("applies"):
        return False
    if float(result.get("confidence", 0)) < threshold:
        return False

    req_quote  = (result.get("requirement_quote") or "").strip()
    viol_quote = (result.get("violation_quote")    or "").strip()
    rec        = (result.get("recommendation")     or "").strip()

    if not req_quote or not viol_quote or len(rec) < 10:
        return False

    evidence_text = f"{req_quote} {viol_quote}".lower()
    if any(phrase in evidence_text for phrase in _VAGUE_JUDGE_PHRASES):
        return False

    def _word_overlap(quote: str, target: str) -> float:
        q_words = set(re.findall(r"\w+", quote.lower())) - _STOPWORDS
        if not q_words:
            return 0.0
        t_words = set(re.findall(r"\w+", target.lower()))
        return len(q_words & t_words) / len(q_words)

    task_r      = task.lower()
    code_r      = code.lower()
    reasoning_r = reasoning.lower()

    req_in_context = (
        req_quote.lower() in task_r
        or req_quote.lower() in reasoning_r
        or _word_overlap(req_quote, task_r + " " + reasoning_r) >= 0.70
    )
    viol_in_context = (
        viol_quote.lower() in code_r
        or viol_quote.lower() in reasoning_r
        or _word_overlap(viol_quote, code_r + " " + reasoning_r) >= 0.70
    )

    return req_in_context and viol_in_context


# ── Helper: extract last python_exec code from a trace ───────────────────────

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


def _parse_json_first(text: str) -> dict:
    """Extract and parse the first complete {...} object from text."""
    import json as _json
    start = text.find("{")
    if start < 0:
        return {}
    depth, end = 0, start
    for i, ch in enumerate(text[start:]):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = start + i + 1
                break
    return _json.loads(text[start:end])


# ── FailureMotif ──────────────────────────────────────────────────────────────

@dataclass
class FailureMotif:
    """A learned logical failure concept — not tied to any specific task or answer."""
    id:                  str
    name:                str
    description:         str
    required_condition:  str
    violation_condition: str
    recommendation:      str
    examples:            list[str] = field(default_factory=list)
    source:              str       = "learned"


# ── MotifStore ────────────────────────────────────────────────────────────────

class MotifStore:
    """Stores learned motifs and retrieves candidates by embedding similarity.

    Retrieval is high-recall (top-k with a minimum similarity floor).
    The applicability judge, not retrieval score, decides whether to fire.
    """

    def __init__(self, embedder: Callable):
        self._embedder = embedder
        self._motifs:  list[FailureMotif]                    = []
        self._vecs:    list[tuple[np.ndarray, FailureMotif]] = []
        self._lock     = threading.Lock()

    def add(self, motif: FailureMotif, signal_text: str = "") -> None:
        embed_text = signal_text or f"{motif.description} {motif.recommendation}"
        vec = np.asarray(self._embedder([embed_text[:600]])[0], dtype='float32')
        norm = np.linalg.norm(vec)
        if norm > 1e-9:
            vec /= norm
        with self._lock:
            self._motifs.append(motif)
            self._vecs.append((vec, motif))

    def update(self, update_id: str, motif: FailureMotif) -> bool:
        """Replace an existing motif and re-embed with updated description."""
        embed_text = f"{motif.description} {motif.recommendation}"
        vec = np.asarray(self._embedder([embed_text[:600]])[0], dtype='float32')
        norm = np.linalg.norm(vec)
        if norm > 1e-9:
            vec /= norm
        with self._lock:
            for i, existing in enumerate(self._motifs):
                if existing.id == update_id:
                    self._motifs[i] = motif
                    self._vecs = [(v, m) for v, m in self._vecs if m.id != update_id]
                    self._vecs.append((vec, motif))
                    return True
        return False

    def retrieve(
        self, task: str, reasoning: str, code: str, top_k: int = 4,
    ) -> list[FailureMotif]:
        """Return top-k motifs by embedding similarity; min cosine similarity 0.35."""
        with self._lock:
            vecs = list(self._vecs)
        if not vecs:
            return []
        query = f"{task}\n{reasoning[-600:]}\n{code[:400]}"
        qvec = np.asarray(self._embedder([query])[0], dtype='float32')
        norm = np.linalg.norm(qvec)
        if norm > 1e-9:
            qvec /= norm
        scored = sorted(
            ((float(np.dot(qvec, v)), m) for v, m in vecs),
            reverse=True,
        )
        return [m for sim, m in scored[:top_k] if sim >= 0.35]

    @property
    def motifs(self) -> list[FailureMotif]:
        with self._lock:
            return list(self._motifs)

    @property
    def count(self) -> int:
        with self._lock:
            return len(self._motifs)


# ── Fire message ──────────────────────────────────────────────────────────────

def _make_fire_message(result: dict, motif_name: str = "") -> str:
    req_quote  = result.get("requirement_quote", "")
    viol_quote = result.get("violation_quote", "")
    explanation = result.get("explanation", result.get("motif_match_explanation", ""))
    recommendation = result.get("recommendation", "")
    header = f"Learned pattern: {motif_name}" if motif_name else "Learned failure pattern"
    lines = [
        f"⚠️ BRAIN:\nSTOP: The monitor detected a likely logical failure before execution.\n",
        f"Evidence ({header}):",
        f"  - Requirement: {req_quote}",
        f"  - Violation:   {viol_quote}",
    ]
    if explanation:
        lines.append(f"  - Explanation: {explanation}")
    lines.append(f"\nRequired correction:\n  {recommendation}")
    lines.append("\nRevise the code before calling the tool again.")
    return "\n".join(lines)


# ── BrainAgent ────────────────────────────────────────────────────────────────

class BrainAgent:
    """Inference-time failure detector using learned-motif memory.

    Usage::
        brain = BrainAgent(embedder)
        agent = tool_agent(["python_exec"])
        agent.monitor = brain

        brain.set_task(idx, task=task_prompt)
        brain.reset()
        trace, tokens = agent(prompt)
        brain.store(trace, label, failure_reason)

    Hooks called by the agent:
        push(text)                       → accumulate reasoning
        before_tool_call(name, inp)      → pre-execution; may return STOP message
        on_tool_call(name, inp, result)  → reactive; stall detection only
    """

    def __init__(
        self, embedder: Callable, k: int = 4, threshold: float = 0.80,
        # legacy kwargs accepted but ignored
        check_interval: float = 1.0, min_chars: int = 200,
    ):
        self._embedder      = embedder
        self._k             = k
        self._fire_threshold = threshold
        self._motif_store   = MotifStore(embedder)

        # Per-task state
        self._reasoning:       list[str] = []
        self._current_task_str: str      = ""
        self._task_idx:         int      = 0
        self._interventions:    int      = 0
        self._turn_count:       int      = 0
        self._stall_streak:     int      = 0

        # Audit
        self.last_fire:    dict | None = None
        self.last_warning: str  | None = None

    # ── task lifecycle ────────────────────────────────────────────────────────

    def set_task(
        self, task_idx: int, probe_fn: Callable | None = None, task: str = "",
    ) -> None:
        self._task_idx = task_idx
        if task:
            self._current_task_str = task

    def reset(self) -> None:
        self._reasoning      = []
        self.last_fire       = None
        self.last_warning    = None
        self._interventions  = 0
        self._turn_count     = 0
        self._stall_streak   = 0

    # ── reasoning accumulation ────────────────────────────────────────────────

    def push(self, text: str) -> None:
        """Receive a reasoning chunk. Accumulates the agent's live reasoning trace."""
        t = text.strip()
        if t:
            self._reasoning.append(t)

    # ── before_tool_call: pre-execution prediction ────────────────────────────

    def before_tool_call(self, name: str, input_dict: dict) -> str | None:
        """Called BEFORE a tool executes. Returns a STOP message or None.

        For each candidate learned motif:
          1. Run applicability judge (haiku LLM call)
          2. Validate judge output deterministically (_validate_judge_result)
          3. If proof is concrete and grounded → fire with STOP message

        The recent reasoning trace is passed to the judge so it can detect
        logical drift, missing assumptions, and plan-code contradictions.
        Trajectory contributes as evidence text, not as a numeric trigger.
        """
        if name != "python_exec" or self._interventions >= 2:
            return None

        code = (input_dict or {}).get("code", "")
        if not code.strip():
            return None

        reasoning = "\n".join(self._reasoning[-20:])
        candidates = self._motif_store.retrieve(
            task=self._current_task_str, reasoning=reasoning,
            code=code, top_k=self._k,
        )

        for motif in candidates:
            result = self._call_judge(motif, code, self._current_task_str, reasoning)
            if result is None:
                continue
            if _validate_judge_result(
                result, self._current_task_str, code, reasoning, self._fire_threshold
            ):
                self._record_fire(result, motif)
                return _make_fire_message(result, motif.name)
            else:
                print(f"[BRAIN JUDGE REJECTED] motif={motif.id!r} "
                      f"req={result.get('requirement_quote','')[:50]!r} "
                      f"viol={result.get('violation_quote','')[:50]!r}")
        return None

    # ── on_tool_call: reactive stall detection ────────────────────────────────

    def on_tool_call(self, name: str, input_dict: dict, result: str) -> str | None:
        """Post-execution hook. Handles stall detection only.

        Motif learning happens in store() after the full task completes,
        not reactively mid-task, to avoid storing recovery patterns as failures.
        """
        self._turn_count += 1
        code = (input_dict or {}).get("code", "") if name == "python_exec" else ""
        is_empty = not result or result.strip() in ("", "(no output)", "None")
        no_input = not code.strip() if name == "python_exec" else (
            not any(str(v).strip() for v in (input_dict or {}).values())
        )

        if no_input or is_empty:
            self._stall_streak += 1
            if self._stall_streak >= 2 and self._interventions < 2:
                self._interventions += 1
                msg = (
                    f"[BRAIN — STALL after {self._stall_streak} unproductive calls]\n"
                    f"Stop repeating the same empty call. Try a completely different approach."
                )
                self.last_warning = msg
                return f"{result}\n\n{msg}" if result else msg
        else:
            self._stall_streak = 0
        return None

    # ── store: post-task motif learning ──────────────────────────────────────

    def store(self, trace: str, label: int, metadata: str = "") -> None:
        """Called after a completed run. Extracts/updates a motif on failure.

        Store first-attempt traces with first-attempt labels — not retry traces.
        Only extracts when the task genuinely failed (label=0) with a reason.
        """
        if label != 0 or not metadata:
            return
        code = _extract_code_from_trace(trace)
        if not code.strip():
            return
        reasoning = "\n".join(self._reasoning[-20:])
        t = threading.Thread(
            target=self._extract_motif,
            args=(code, self._current_task_str, reasoning, metadata),
            daemon=True,
            name="brain-extract",
        )
        t.start()
        t.join(timeout=6.0)

    # ── internals ─────────────────────────────────────────────────────────────

    def _call_judge(
        self, motif: FailureMotif, code: str, task: str, reasoning: str,
    ) -> dict | None:
        """Ask haiku whether the motif concretely applies. Returns parsed JSON or None."""
        try:
            from .agents import _anthropic_call
        except ImportError:
            return None

        prompt = _APPLICABILITY_JUDGE_PROMPT.format(
            task                 = task[:500]       or "(not provided)",
            reasoning            = reasoning[-900:] if reasoning else "(no reasoning yet)",
            code                 = code[:700],
            motif_name           = motif.name,
            motif_description    = motif.description,
            required_condition   = motif.required_condition  or motif.description,
            violation_condition  = motif.violation_condition or "(see description)",
            motif_recommendation = motif.recommendation,
        )
        try:
            from .agents import _anthropic_call
            text, _ = _anthropic_call("claude-haiku-4-5-20251001", prompt, max_tokens=320)
            text = text.strip()
            if text.startswith("```"):
                text = re.sub(r"^```[a-z]*\n?", "", text)
                text = re.sub(r"\n?```$", "", text)
            result = _parse_json_first(text)
            print(f"[BRAIN JUDGE RAW] motif={motif.id!r} applies={result.get('applies')} "
                  f"conf={result.get('confidence', 0):.2f} "
                  f"req={result.get('requirement_quote','')[:50]!r} "
                  f"viol={result.get('violation_quote','')[:50]!r}")
            return result
        except Exception as e:
            print(f"[BRAIN JUDGE ERROR] motif={motif.id!r} err={e!r}")
            return None

    def _extract_motif(
        self, code: str, task: str, reasoning: str, reason: str,
    ) -> None:
        """LLM call to extract or update a learned motif from a failure."""
        try:
            from .agents import _anthropic_call
        except ImportError:
            return

        existing = self._motif_store.motifs
        existing_lines = (
            "\n".join(f"  id={m.id!r}  description={m.description!r}" for m in existing)
            if existing else "  (none yet)"
        )
        prompt = _EXTRACT_MOTIF_PROMPT.format(
            task     = task[:400]     or "(not provided)",
            reasoning= reasoning[-600:] or "(none)",
            code     = code[:800],
            reason   = reason[:300]   or "(not provided)",
            existing = existing_lines,
        )
        try:
            text, _ = _anthropic_call("claude-haiku-4-5-20251001", prompt, max_tokens=400)
            text = text.strip()
            if text.startswith("```"):
                text = re.sub(r"^```[a-z]*\n?", "", text)
                text = re.sub(r"\n?```$", "", text)
            data = _parse_json_first(text)
        except Exception:
            return

        if not data:
            return

        desc = str(data.get("description", "")).strip()
        if not desc:
            return

        _vague = {"incomplete", "truncation", "truncated", "partial",
                  "placeholder", "stub", "unfinished", "todo"}
        if any(w in desc.lower() for w in _vague):
            return

        update_id = data.get("update_id") or None
        motif = FailureMotif(
            id                  = str(data.get("id",   "learned"))[:40],
            name                = str(data.get("name", "Learned motif"))[:80],
            description         = desc[:250],
            required_condition  = str(data.get("required_condition",  "")).strip()[:300],
            violation_condition = str(data.get("violation_condition", "")).strip()[:300],
            recommendation      = str(data.get("recommendation",      "")).strip()[:300],
            examples            = [str(e) for e in data.get("examples", [])[:3]],
            source              = "learned",
        )

        if update_id:
            existing_motif = next(
                (m for m in self._motif_store.motifs if m.id == update_id), None
            )
            if existing_motif:
                updated = FailureMotif(
                    id                  = existing_motif.id,
                    name                = existing_motif.name,
                    description         = motif.description or existing_motif.description,
                    required_condition  = motif.required_condition  or existing_motif.required_condition,
                    violation_condition = motif.violation_condition or existing_motif.violation_condition,
                    recommendation      = motif.recommendation      or existing_motif.recommendation,
                    examples            = list({*existing_motif.examples, *motif.examples})[:5],
                    source              = "learned",
                )
                self._motif_store.update(update_id, updated)
                print(f"[BRAIN MOTIF UPDATED] {updated.id}: {updated.description!r}")
                return

        self._motif_store.add(motif)
        print(f"[BRAIN MOTIF LEARNED] {motif.id}: {motif.description!r}")

    def _record_fire(self, result: dict, motif: FailureMotif) -> None:
        self._interventions += 1
        self.last_fire = {
            "task":               self._task_idx,
            "hook":               "before_tool_call",
            "motif":              motif.id,
            "confidence":         result.get("confidence"),
            "requirement_quote":  result.get("requirement_quote", ""),
            "violation_quote":    result.get("violation_quote", ""),
        }
        self.last_warning = _make_fire_message(result, motif.name)
        print(f"[BRAIN JUDGE ACCEPTED] motif={motif.id!r} conf={result.get('confidence',0):.2f} "
              f"req={result.get('requirement_quote','')[:60]!r} "
              f"viol={result.get('violation_quote','')[:60]!r}")
        print(f"[BRAIN FIRE] {self.last_fire}")

    # ── backward-compatible properties ────────────────────────────────────────

    @property
    def n_stored(self) -> int:
        return self._motif_store.count

    @property
    def n_fail(self) -> int:
        return 0

    @property
    def n_pass(self) -> int:
        return 0

    def get_trajectory(self) -> list:
        return []

    def predict_pre_generation(
        self, prompt: str, use_llm: bool = True,
    ) -> tuple[None, None]:
        return None, None

    def seed(self, items: list[dict]) -> None:
        pass

    def wrap_any(self, agent_or_model, retry_fn=None, cot: bool = False):
        return agent_or_model

    @property
    def failure_store(self):
        return self

    def all_vecs(self):
        return None, []

    # Legacy attribute aliases used by use.py
    @property
    def _live_steps(self) -> list:
        return self._reasoning

    @property
    def _traj_store(self):
        return self

    @property
    def _threshold(self) -> float:
        return self._fire_threshold

    def on_chunk(self, text: str) -> None:
        pass

    def pulse(self) -> bool:
        return False

    @property
    def should_bail(self) -> bool:
        return False

    def get_intervention(self) -> str | None:
        return self.last_warning
