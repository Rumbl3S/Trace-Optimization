"""Offline tests for BrainAgent and _validate_judge_result.

All tests use a stub embedder — no API keys needed.
"""
from __future__ import annotations

import numpy as np
import pytest

from trace_use.brain import (
    BrainAgent,
    FailureMotif,
    MotifStore,
    _validate_judge_result,
    _make_fire_message,
)


# ── Stub embedder ─────────────────────────────────────────────────────────────

def _stub_embedder(texts):
    """Fixed unit-vectors keyed by text — identical inputs → identical vectors."""
    rng = np.random.default_rng(seed=sum(ord(c) for c in (texts[0] or "x")))
    n   = len(texts)
    v   = rng.standard_normal((n, 64)).astype("float32")
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    return v / (norms + 1e-9)


def _make_brain(threshold: float = 0.80) -> BrainAgent:
    return BrainAgent(_stub_embedder, k=4, threshold=threshold)


def _make_motif(**kwargs) -> FailureMotif:
    defaults = dict(
        id="test_motif",
        name="Test Motif",
        description="some logical failure description",
        required_condition="task requires tiebreak on equal keys",
        violation_condition="sort uses single key with no tuple",
        recommendation="use tuple key for primary and secondary sort criteria",
        examples=[],
        source="learned",
    )
    defaults.update(kwargs)
    return FailureMotif(**defaults)


# ── MotifStore tests ──────────────────────────────────────────────────────────

def test_motif_store_add_and_count():
    store = MotifStore(_stub_embedder)
    assert store.count == 0
    store.add(_make_motif(id="m1"), signal_text="sort tiebreak missing")
    assert store.count == 1
    store.add(_make_motif(id="m2"), signal_text="wrong formula used")
    assert store.count == 2


def test_motif_store_retrieve_returns_motifs():
    store = MotifStore(_stub_embedder)
    m = _make_motif(id="m1")
    store.add(m, signal_text="sort tiebreak missing")
    results = store.retrieve("sort users by score then name", "", "", top_k=4)
    # May or may not match above 0.35, but must return a list
    assert isinstance(results, list)
    assert all(isinstance(r, FailureMotif) for r in results)


def test_motif_store_update_replaces_motif():
    store = MotifStore(_stub_embedder)
    m = _make_motif(id="m1", description="original description")
    store.add(m)
    updated = _make_motif(id="m1", description="updated description after second occurrence")
    result = store.update("m1", updated)
    assert result is True
    assert store.count == 1
    assert store.motifs[0].description == "updated description after second occurrence"


def test_motif_store_update_nonexistent_returns_false():
    store = MotifStore(_stub_embedder)
    result = store.update("does_not_exist", _make_motif())
    assert result is False


# ── BrainAgent lifecycle ──────────────────────────────────────────────────────

def test_brain_set_task_stores_task_string():
    brain = _make_brain()
    brain.set_task(3, task="Sort users by score descending; break ties by name")
    assert brain._current_task_str == "Sort users by score descending; break ties by name"
    assert brain._task_idx == 3


def test_brain_reset_clears_reasoning():
    brain = _make_brain()
    brain.push("step 1 reasoning")
    brain.push("step 2 more reasoning")
    assert len(brain._reasoning) == 2
    brain.reset()
    assert brain._reasoning == []
    assert brain.last_fire is None


def test_brain_push_accumulates():
    brain = _make_brain()
    brain.push("I will sort by score.")
    brain.push("Now I'll add a tiebreak.")
    assert len(brain._reasoning) == 2
    assert brain._reasoning[0] == "I will sort by score."


# ── before_tool_call with empty store → None ─────────────────────────────────

def test_before_tool_call_empty_store_returns_none():
    brain = _make_brain()
    brain.set_task(0, task="Sort users by score")
    brain._current_task_str = "Sort users by score"
    result = brain.before_tool_call("python_exec", {"code": "sorted(users, key=lambda u: u['score'], reverse=True)"})
    assert result is None


def test_before_tool_call_non_python_exec_returns_none():
    brain = _make_brain()
    brain._current_task_str = "Sort users"
    result = brain.before_tool_call("bash", {"command": "ls -la"})
    assert result is None


def test_before_tool_call_empty_code_returns_none():
    brain = _make_brain()
    brain._current_task_str = "Sort users"
    result = brain.before_tool_call("python_exec", {"code": ""})
    assert result is None


# ── before_tool_call fires when judge returns concrete proof (monkeypatch) ────

def test_before_tool_call_fires_on_concrete_judge_proof():
    """Monkeypatch retrieve+_call_judge to return valid proof — brain must fire."""
    motif = _make_motif(id="secondary_sort_missing")
    brain = _make_brain(threshold=0.80)
    brain._current_task_str = "Sort users by score descending; break ties alphabetically by name"

    # Bypass embedding similarity check — guarantee the motif is a candidate
    brain._motif_store.retrieve = lambda **kwargs: [motif]

    concrete_result = {
        "applies": True,
        "confidence": 0.92,
        "requirement_quote": "break ties alphabetically by name",
        "violation_quote":   "sorted(users, key=lambda u: -u['score'])",
        "explanation":       "code uses single key, ignoring the tiebreak requirement",
        "recommendation":    "key=lambda u: (-u['score'], u['name'])",
    }
    brain._call_judge = lambda *args, **kwargs: concrete_result

    code = "sorted(users, key=lambda u: -u['score'])"
    result = brain.before_tool_call("python_exec", {"code": code})
    assert result is not None, "Brain did not fire despite concrete judge proof"
    assert "STOP" in result
    assert brain.last_fire is not None


def test_before_tool_call_no_fire_when_judge_returns_empty_quotes():
    """Judge returns applies=True but empty quotes → must NOT fire."""
    motif = _make_motif(id="secondary_sort_missing")
    brain = _make_brain(threshold=0.50)
    brain._current_task_str = "Sort users by score"
    brain._motif_store.retrieve = lambda **kwargs: [motif]

    vague_result = {
        "applies": True,
        "confidence": 0.95,
        "requirement_quote": "",   # no proof
        "violation_quote":   "",   # no proof
        "explanation":       "task implies tiebreak might be needed",
        "recommendation":    "add secondary sort key",
    }
    brain._call_judge = lambda *args, **kwargs: vague_result

    code = "sorted(users, key=lambda u: -u['score'])"
    result = brain.before_tool_call("python_exec", {"code": code})
    assert result is None, f"Empty-quote judge must not fire: {result!r}"


def test_before_tool_call_no_fire_when_confidence_below_threshold():
    """Judge returns applies=True, conf=0.60 — below 0.80 → must NOT fire."""
    motif = _make_motif(id="secondary_sort_missing")
    brain = _make_brain(threshold=0.80)
    brain._current_task_str = "Sort users by score descending; break ties by name"
    brain._motif_store.retrieve = lambda **kwargs: [motif]

    low_conf_result = {
        "applies": True,
        "confidence": 0.60,
        "requirement_quote": "break ties by name",
        "violation_quote":   "sorted(users, key=lambda u: -u['score'])",
        "explanation":       "single key sort",
        "recommendation":    "add name as secondary sort key",
    }
    brain._call_judge = lambda *args, **kwargs: low_conf_result

    code = "sorted(users, key=lambda u: -u['score'])"
    result = brain.before_tool_call("python_exec", {"code": code})
    assert result is None, f"Low-confidence judge must not fire: {result!r}"


def test_before_tool_call_no_fire_when_judge_returns_applies_false():
    """Judge returns applies=False → must NOT fire regardless."""
    motif = _make_motif(id="secondary_sort_missing")
    brain = _make_brain(threshold=0.50)
    brain._current_task_str = "Sort users by score"
    brain._motif_store.retrieve = lambda **kwargs: [motif]

    not_applicable = {
        "applies": False,
        "confidence": 0.95,
        "requirement_quote": "",
        "violation_quote":   "",
        "explanation":       "task has no tiebreak requirement",
        "recommendation":    "",
    }
    brain._call_judge = lambda *args, **kwargs: not_applicable

    code = "sorted(users, key=lambda u: -u['score'])"
    result = brain.before_tool_call("python_exec", {"code": code})
    assert result is None, f"applies=False must never fire: {result!r}"


# ── on_tool_call stall detection ──────────────────────────────────────────────

def test_on_tool_call_no_warning_on_successful_result():
    brain = _make_brain()
    brain._current_task_str = "Sort users"
    result = brain.on_tool_call("python_exec", {"code": "print(sorted(x))"}, "output: [1,2,3]")
    # Successful result should not stall
    assert brain._stall_streak == 0
    assert result is None


def test_on_tool_call_stall_warning_after_two_empty_calls():
    brain = _make_brain()
    brain._current_task_str = "Sort users"
    code = "print('hello')"
    brain.on_tool_call("python_exec", {"code": code}, "")
    brain.on_tool_call("python_exec", {"code": code}, "")
    result = brain.on_tool_call("python_exec", {"code": code}, "")
    assert result is not None
    assert "BRAIN" in result or "stall" in result.lower()


def test_on_tool_call_stall_resets_on_real_output():
    brain = _make_brain()
    brain._current_task_str = "Sort users"
    code = "print(x)"
    brain.on_tool_call("python_exec", {"code": code}, "")
    brain.on_tool_call("python_exec", {"code": code}, "[1, 2, 3]")
    assert brain._stall_streak == 0


# ── store: pass label must not extract motif ─────────────────────────────────

def test_store_pass_label_never_extracts_motif():
    """store() called with label=1 (pass) must not trigger motif extraction."""
    brain = _make_brain()
    brain._current_task_str = "Sort users"
    brain.store("some trace [tool:python_exec({\"code\": \"x = 1\"})]", label=1, metadata="wrong formula")
    # Because label=1 → nothing to extract → store count must be 0
    assert brain._motif_store.count == 0


def test_store_fail_no_code_in_trace_does_not_crash():
    """store() with label=0 but no extractable code must not crash."""
    brain = _make_brain()
    brain._current_task_str = "Sort users"
    brain.store("trace with no tool calls", label=0, metadata="failed")
    # No crash, no motif (code extraction returns empty)
    assert brain._motif_store.count == 0


# ── Backward-compat properties ────────────────────────────────────────────────

def test_compat_n_stored_is_motif_count():
    brain = _make_brain()
    assert brain.n_stored == 0
    brain._motif_store.add(_make_motif(id="m1"))
    assert brain.n_stored == 1


def test_compat_n_pass_and_n_fail_return_zero():
    brain = _make_brain()
    assert brain.n_pass == 0
    assert brain.n_fail == 0


def test_compat_predict_pre_generation_returns_none_pair():
    brain = _make_brain()
    result = brain.predict_pre_generation("some prompt", use_llm=True)
    assert result == (None, None)


def test_compat_seed_does_not_crash():
    brain = _make_brain()
    brain.seed([{"trace": "x", "label": 0, "metadata": "y"}])
    # no-op — must not crash
    assert brain.n_stored == 0


def test_compat_traj_store_threshold_attribute():
    """use.py accesses brain._traj_store._threshold directly."""
    brain = _make_brain(threshold=0.80)
    assert brain._traj_store._threshold == 0.80


def test_compat_current_task_str_direct_access():
    """use.py sets brain._current_task_str directly."""
    brain = _make_brain()
    brain._current_task_str = "Direct task assignment from use.py"
    assert brain._current_task_str == "Direct task assignment from use.py"


def test_compat_failure_store_returns_something():
    """use.py accesses brain.failure_store — must not crash."""
    brain = _make_brain()
    fs = brain.failure_store
    assert fs is not None


def test_compat_get_trajectory_returns_list():
    brain = _make_brain()
    assert brain.get_trajectory() == []


def test_compat_pulse_returns_false():
    brain = _make_brain()
    assert brain.pulse() is False


def test_compat_on_chunk_returns_none():
    brain = _make_brain()
    result = brain.on_chunk("some reasoning text")
    assert result is None


# ── _validate_judge_result tests (F–K) ───────────────────────────────────────

# F: missing-data motif must NOT fire when task says all fields are present
def test_F_missing_data_motif_no_fire_when_all_fields_present():
    """Vague 'task implies' phrase in requirement_quote must be rejected."""
    task      = "Each user has name and score; rank users by score"
    code      = "sorted(users, key=lambda u: u['score'], reverse=True)"
    reasoning = "I will sort users by score field directly."

    result_vague = {
        "applies": True,
        "confidence": 0.80,
        "requirement_quote": "task implies required fields may be missing",
        "violation_quote":   "u['score'] will raise KeyError if missing",
        "explanation":       "possible missing key",
        "recommendation":    "use .get('score', 0) to be safe",
    }
    assert not _validate_judge_result(result_vague, task, code, reasoning, threshold=0.50), (
        "Vague 'task implies' phrase must be rejected by _validate_judge_result"
    )


# G: concrete, grounded evidence must be accepted
def test_G_missing_data_motif_fires_with_explicit_requirement():
    """Concrete quotes grounded in actual task and code text must be accepted."""
    task      = "Process records. Raise ValueError if the 'email' field is missing."
    code      = "email = record.get('email', '')\nsend_email(email)"
    reasoning = "I will use .get to retrieve the email field with a default."

    result_concrete = {
        "applies": True,
        "confidence": 0.85,
        "requirement_quote": "Raise ValueError if the 'email' field is missing",
        "violation_quote":   "email = record.get('email', '')",
        "explanation":       "code swallows missing field instead of raising",
        "recommendation":    "if 'email' not in record: raise ValueError('email missing')",
    }
    assert _validate_judge_result(result_concrete, task, code, reasoning, threshold=0.50), (
        "Concrete, grounded proof must be accepted by _validate_judge_result"
    )


# H: secondary-sort motif must NOT fire when task has no tiebreak requirement
def test_H_secondary_sort_no_fire_without_tiebreak_requirement():
    """Empty requirement_quote when task has no tiebreak → must be rejected."""
    task      = "Sort users by name"
    code      = "sorted(users, key=lambda u: u['name'])"
    reasoning = "Sort alphabetically by name field."

    result_no_req = {
        "applies": True,
        "confidence": 0.75,
        "requirement_quote": "",    # nothing to quote — task has no tiebreak
        "violation_quote":   "sorted(users, key=lambda u: u['name'])",
        "explanation":       "missing secondary key",
        "recommendation":    "add tiebreak key",
    }
    assert not _validate_judge_result(result_no_req, task, code, reasoning, threshold=0.50), (
        "Empty requirement_quote must be rejected"
    )


# I: secondary-sort motif DOES fire when task explicitly requires tiebreak
def test_I_secondary_sort_fires_with_explicit_tiebreak():
    """Both quotes grounded in actual task/code text must pass."""
    task      = "Sort users by score descending; break ties alphabetically by name"
    code      = "sorted(users, key=lambda u: -u['score'])"
    reasoning = "I'll sort by score in descending order using a negative key."

    result_concrete = {
        "applies": True,
        "confidence": 0.88,
        "requirement_quote": "break ties alphabetically by name",
        "violation_quote":   "sorted(users, key=lambda u: -u['score'])",
        "explanation":       "single-key sort ignores alphabetical tiebreak",
        "recommendation":    "key=lambda u: (-u['score'], u['name'])",
    }
    assert _validate_judge_result(result_concrete, task, code, reasoning, threshold=0.50), (
        "Explicit tiebreak requirement with matching violation must be accepted"
    )


# J: proof-free judge must not fire even with motif present
def test_J_proof_free_judge_cannot_fire():
    """before_tool_call must not fire when _call_judge returns empty quotes.

    The retrieve path is monkeypatched so the motif IS a candidate.
    Only the proof gate (_validate_judge_result) should block the fire.
    """
    motif = _make_motif(id="secondary_sort_missing")
    brain = _make_brain(threshold=0.50)
    brain._current_task_str = "Sort users by name"
    brain._motif_store.retrieve = lambda **kwargs: [motif]

    # Judge returns applies=True but with completely empty evidence
    brain._call_judge = lambda *args, **kwargs: {
        "applies": True,
        "confidence": 0.95,
        "requirement_quote": "",
        "violation_quote":   "",
        "explanation":       "task implies this might apply",
        "recommendation":    "check",
    }

    code = "sorted(users, key=lambda u: u['name'])"
    result = brain.before_tool_call("python_exec", {"code": code})
    assert result is None, (
        f"Proof-free judge result must not fire: {result!r}"
    )


# K: all listed vague phrases in evidence text must cause rejection
def test_K_vague_judge_phrases_rejected():
    """Any vague speculative phrase in either quote → rejected."""
    task      = "Process items and return results"
    code      = "return [process(x) for x in items]"
    reasoning = "I will iterate and process each item."

    vague_phrases = [
        "task implies",
        "likely",
        "may need",
        "could fail",
        "should consider",
        "similar to",
        "general requirement",
        "might",
        "possibly",
        "perhaps",
        "seems like",
        "probably",
    ]
    for phrase in vague_phrases:
        result = {
            "applies": True,
            "confidence": 0.90,
            "requirement_quote": f"The task {phrase} require validation here",
            "violation_quote":   "return [process(x) for x in items]",
            "explanation":       "found a match",
            "recommendation":    "add validation before processing",
        }
        assert not _validate_judge_result(result, task, code, reasoning, threshold=0.50), (
            f"Vague phrase {phrase!r} in evidence must be rejected"
        )
