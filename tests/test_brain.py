"""Offline tests for the evidence-based BrainAgent fire system.

All tests use a stub embedder — no API keys needed.
"""
from __future__ import annotations

import numpy as np
import pytest

from trace_use.brain import (
    BrainAgent,
    BrainEvidence,
    ConstraintChecker,
    FailureMotif,
    MotifStore,
    PlanCodeMismatchChecker,
    _CONCRETE_EVIDENCE_KINDS,
    make_intervention,
)
from trace_use.brain import LogicalFailureStore, LatentStep


# ── Stub embedder ─────────────────────────────────────────────────────────────

def _stub_embedder(texts):
    """Returns a fixed unit-vector so no real API call is made."""
    rng = np.random.default_rng(42)
    n   = len(texts)
    v   = rng.standard_normal((n, 64)).astype("float32")
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    return v / (norms + 1e-9)


def _make_brain(threshold: float = 0.45, **kwargs) -> BrainAgent:
    return BrainAgent(_stub_embedder, k=3, threshold=threshold, **kwargs)


def _step(text: str) -> LatentStep:
    vec = _stub_embedder([text])[0]
    return LatentStep(vec=vec, drift=0.1, step_type="reason", index=0, text=text)


# ── 1. Correct Sharpe code never fires ───────────────────────────────────────

def test_correct_sharpe_code_no_fire():
    brain = _make_brain()
    brain.seed_motifs([
        {
            "id": "additive_rate_conversion",
            "surface_pattern": r"(rf|risk_free|rate)\s*[^(=\n]*?/\s*252",
            "neg_pattern": r"\(1\s*\+.*?\)\s*\*\*|\bexp\s*\(",
            "task_keywords": ["sharpe"],
            "confidence": 0.88,
        }
    ])
    brain._current_task_str = "Compute Sharpe ratio."
    correct_code = (
        "rf_daily = (1 + risk_free_annual)**(1/252) - 1\n"
        "excess = [r - rf_daily for r in returns]\n"
        "return (np.mean(excess) / np.std(excess, ddof=0)) * np.sqrt(252)\n"
    )
    result = brain.before_tool_call("python_exec", {"code": correct_code})
    assert result is None, f"Brain fired on correct code: {result}"


# ── 2. Additive rf motif fires ────────────────────────────────────────────────

def test_additive_rf_motif_fires():
    brain = _make_brain()
    brain.seed_motifs([
        {
            "id": "additive_rate_conversion",
            "surface_pattern": r"(rf|risk_free|rate)\s*[^(=\n]*?/\s*252",
            "neg_pattern": r"\(1\s*\+.*?\)\s*\*\*|\bexp\s*\(",
            "task_keywords": ["sharpe"],
            "confidence": 0.88,
        }
    ])
    brain._current_task_str = "Compute Sharpe ratio with risk-free rate."
    wrong_code = (
        "rf_daily = risk_free_annual / 252\n"
        "excess = [r - rf_daily for r in returns]\n"
        "return np.mean(excess) / np.std(excess) * np.sqrt(252)\n"
    )
    result = brain.before_tool_call("python_exec", {"code": wrong_code})
    assert result is not None, "Brain did not fire on additive rf/252"
    assert "BRAIN" in result


# ── 3. Generic reasoning step alone never fires ───────────────────────────────

def test_generic_reasoning_no_fire():
    brain = _make_brain()
    brain.seed_motifs([
        {
            "id": "additive_rate_conversion",
            "surface_pattern": r"rf\s*/\s*252",
            "neg_pattern": r"\(1\s*\+.*?\)\s*\*\*",
            "task_keywords": ["sharpe"],
            "confidence": 0.88,
        }
    ])
    brain._current_task_str = "Compute portfolio metrics."
    # First reasoning step — no code submitted yet, no tool call
    brain._live_steps.append(_step("I'll start by defining the data structure."))
    result = brain.before_tool_call("python_exec", {"code": ""})
    assert result is None


# ── 4. Plan-code mismatch fires ───────────────────────────────────────────────

def test_plan_code_mismatch_fires():
    brain = _make_brain()
    brain._current_task_str = "Compute Sharpe ratio."
    # Reasoning says compound, code uses additive
    brain._live_steps = [
        _step("I need to use compound daily rate conversion from the annual rf."),
    ]
    wrong_code = "rf_daily = risk_free_annual / 252\nreturn (mu - rf_daily) / std * 252**0.5"
    result = brain.before_tool_call("python_exec", {"code": wrong_code})
    assert result is not None, "Plan-code mismatch not detected"
    assert "BRAIN" in result


# ── 5. No mismatch when code matches the plan ────────────────────────────────

def test_no_mismatch_when_code_matches_plan():
    brain = _make_brain()
    brain._current_task_str = "Compute Sharpe ratio."
    brain._live_steps = [
        _step("I need to use compound daily rate conversion from the annual rf."),
    ]
    correct_code = (
        "rf_daily = (1 + risk_free_annual)**(1/252) - 1\n"
        "excess = returns - rf_daily\n"
        "return excess.mean() / excess.std(ddof=0) * np.sqrt(252)"
    )
    result = brain.before_tool_call("python_exec", {"code": correct_code})
    assert result is None, f"False positive on matching plan+code: {result}"


# ── 6. Constraint fires when task EXPLICITLY requires population std ──────────

def test_constraint_fires_when_task_explicit():
    brain = _make_brain()
    brain._current_task_str = (
        "Compute annualised volatility. Use population std (divide by n, not n-1)."
    )
    wrong_code = "vol = np.std(returns, ddof=1) * np.sqrt(252)"
    result = brain.before_tool_call("python_exec", {"code": wrong_code})
    assert result is not None, "ConstraintChecker did not fire on population-std violation"
    assert "BRAIN" in result


# ── 7. Constraint silent when task is vague (no explicit requirement) ─────────

def test_constraint_silent_when_task_vague():
    brain = _make_brain()
    brain._current_task_str = "Compute the volatility of the portfolio returns."
    wrong_code = "vol = np.std(returns, ddof=1) * np.sqrt(252)"
    result = brain.before_tool_call("python_exec", {"code": wrong_code})
    assert result is None, (
        f"ConstraintChecker fired from domain knowledge alone (no explicit constraint): {result}"
    )


# ── 8. Seeded motifs fire with zero real failures in the store ────────────────

def test_seeded_motifs_fire_with_no_real_failures():
    brain = _make_brain()
    brain.seed_motifs([
        {
            "id": "sample_std_for_population",
            "surface_pattern": r"ddof\s*=\s*1|statistics\.stdev\s*\(",
            "neg_pattern": r"ddof\s*=\s*0",
            "task_keywords": ["volatility", "vol", "std"],
            "confidence": 0.92,
        }
    ])
    assert brain.n_stored == 0, "Store should be empty before any real tasks"
    brain._current_task_str = "Compute portfolio volatility."
    wrong_code = "vol = np.std(portfolio_returns, ddof=1) * np.sqrt(252)"
    result = brain.before_tool_call("python_exec", {"code": wrong_code})
    assert result is not None, "Seeded motif did not fire with empty store"


# ── 9. Trajectory kNN alone never fires ──────────────────────────────────────

def test_trajectory_knn_alone_does_not_fire():
    """Store 5 failures via seed(), then call before_tool_call with correct code.
    The trajectory kNN will match (similar stub embeddings), but since it's a
    banned source and the code is clean, the brain must NOT fire.
    """
    brain = _make_brain()
    for i in range(5):
        brain.seed([{
            "trace": f"finance task failure trace {i}",
            "label": 0,
            "metadata": f"Wrong formula {i}",
        }])
    assert brain.n_stored == 5
    brain._current_task_str = "Compute Sharpe ratio."
    correct_code = (
        "rf_daily = (1 + risk_free_annual)**(1/252) - 1\n"
        "excess = returns - rf_daily\n"
        "return float(excess.mean() / excess.std(ddof=0) * np.sqrt(252))"
    )
    result = brain.before_tool_call("python_exec", {"code": correct_code})
    assert result is None, (
        f"Brain fired from trajectory kNN alone (banned source): {result}"
    )


# ── Test A: p_fail alone cannot fire ─────────────────────────────────────────

def test_A_p_fail_alone_cannot_fire():
    """Even with p_fail=1.0 from nearest neighbors, correct code must not fire.

    This directly tests the rule: p_fail may only be metadata, never a trigger.
    """
    brain = _make_brain(threshold=0.10)  # very low threshold to expose false positives
    # Fill store with 10 failures to ensure p_fail is high
    for i in range(10):
        brain.seed([{
            "trace": f"finance risk metric failure {i} " * 10,
            "label": 0,
            "metadata": f"failed task {i}",
        }])
    brain._current_task_str = "Compute Sharpe ratio with compound risk-free rate."
    # Correct code: should never fire regardless of trajectory p_fail
    correct_code = (
        "rf_daily = (1 + risk_free_annual)**(1/252) - 1\n"
        "excess = [r - rf_daily for r in returns]\n"
        "vol = (sum((x - sum(excess)/len(excess))**2 for x in excess)/len(excess))**0.5\n"
        "return (sum(excess)/len(excess)) / vol * 252**0.5"
    )
    result = brain.before_tool_call("python_exec", {"code": correct_code})
    assert result is None, (
        f"p_fail alone triggered a fire — trajectory is context only: {result!r}"
    )


# ── Test B: "Trajectory resembles past failures" message is impossible ─────────

def test_B_banned_message_never_appears():
    """The literal string 'Trajectory resembles past failures' must never be
    returned from any BrainAgent hook.
    """
    brain = _make_brain(threshold=0.10)
    for i in range(8):
        brain.seed([{
            "trace": f"finance failure trace {i} " * 5,
            "label": 0,
            "metadata": f"error {i}",
        }])
    brain.push("I will solve this finance problem.")
    brain._current_task_str = "Compute portfolio risk."

    banned = "Trajectory resembles past failures"

    # before_tool_call
    code = "vol = np.std(returns, ddof=1) * 252**0.5"
    pre = brain.before_tool_call("python_exec", {"code": code})
    if pre is not None:
        assert banned not in pre, f"Banned message in before_tool_call: {pre!r}"

    # pulse
    brain._buffer = "I am generating analysis for this finance task."
    brain.pulse()
    if brain.last_warning:
        assert banned not in brain.last_warning, (
            f"Banned message in pulse last_warning: {brain.last_warning!r}"
        )

    # on_chunk
    chunk_result = brain.on_chunk("I will compute volatility using standard deviation.")
    assert chunk_result is None, (
        f"on_chunk returned non-None (should never fire): {chunk_result!r}"
    )


# ── Test C: concrete motif fires correctly ────────────────────────────────────

def test_C_concrete_motif_fires():
    """Concrete motif match must fire even with no prior stored failures."""
    brain = _make_brain()
    brain.seed_motifs([{
        "id": "additive_rf",
        "name": "Additive rate conversion",
        "description": "rf/252 is wrong when compounding is required",
        "surface_pattern": r"(risk_free|rf)\s*[^(=\n]*?/\s*252",
        "neg_pattern": r"\(1\s*\+.*?\)\s*\*\*|\bexp\s*\(",
        "task_keywords": ["sharpe", "risk_free", "annual"],
        "confidence": 0.88,
        "recommendation": "rf_daily = (1 + rf_annual)**(1/252) - 1",
    }])
    brain._current_task_str = "Compute Sharpe ratio. risk_free_annual is the annual rf."
    wrong_code = "rf_daily = risk_free_annual / 252\nreturn (mu - rf_daily) / sigma"
    result = brain.before_tool_call("python_exec", {"code": wrong_code})
    assert result is not None, "Concrete motif (rf/252) should fire"
    assert "BRAIN" in result
    assert "Trajectory resembles" not in result


# ── Test D: correct code does not fire despite high p_fail ────────────────────

def test_D_correct_code_no_fire_despite_high_p_fail():
    """Finance task with correct code must not fire even when all stored
    trajectories are failures (making p_fail ≈ 1.0).
    """
    brain = _make_brain(threshold=0.05)  # extremely low threshold
    brain.seed_motifs([{
        "id": "additive_rf",
        "name": "Additive rate conversion",
        "description": "rf/252 is wrong",
        "surface_pattern": r"(risk_free|rf)\s*[^(=\n]*?/\s*252",
        "neg_pattern": r"\(1\s*\+.*?\)\s*\*\*|\bexp\s*\(",
        "task_keywords": ["sharpe"],
        "confidence": 0.88,
        "recommendation": "Use compound",
    }])
    # All stored traces are failures
    for i in range(10):
        brain.seed([{
            "trace": f"sharpe ratio failure {i} " * 8,
            "label": 0,
            "metadata": f"used additive conversion {i}",
        }])
    brain._current_task_str = "Compute Sharpe ratio with risk_free_annual."
    # Correct code — neg_pattern should prevent motif from firing
    correct_code = (
        "rf_daily = (1 + risk_free_annual)**(1/252) - 1\n"
        "excess = [r - rf_daily for r in returns]\n"
        "vol = (sum((x - sum(excess)/len(excess))**2 for x in excess) / len(excess))**0.5\n"
        "return sum(excess)/len(excess) / vol * 252**0.5"
    )
    result = brain.before_tool_call("python_exec", {"code": correct_code})
    assert result is None, (
        f"Correct code fired despite high p_fail — domain prior leaked: {result!r}"
    )


# ── Test E: after-tool learned motif requires execution error ─────────────────

def test_trajectory_resets_between_tasks():
    """get_trajectory() must reflect ONLY the current task after reset().

    Regression test: the old reset() did not clear _trajectory, so any(pt.fired)
    would return True for all tasks after the first legitimate fire.
    """
    brain = _make_brain()
    brain.seed_motifs([{
        "id": "ddof1",
        "surface_pattern": r"ddof\s*=\s*1",
        "neg_pattern": "",
        "task_keywords": [],
        "confidence": 0.92,
    }])

    # Task 1: fire correctly (code uses ddof=1)
    brain.set_task(0, task="Compute volatility.")
    brain.reset()
    r1 = brain.before_tool_call("python_exec", {"code": "np.std(x, ddof=1)"})
    assert r1 is not None, "Task 1 should fire"
    assert brain.last_fire is not None
    traj1 = brain.get_trajectory()
    assert any(pt.fired for pt in traj1), "Task 1 trajectory should show fired"

    # Task 2: no fire (correct code)
    brain.set_task(1, task="Compute volatility.")
    brain.reset()   # MUST clear _trajectory
    traj_after_reset = brain.get_trajectory()
    assert traj_after_reset == [], "reset() must clear _trajectory — stale fires must not persist"
    r2 = brain.before_tool_call("python_exec", {"code": "np.std(x, ddof=0)"})
    assert r2 is None, "Task 2 should not fire (correct ddof=0)"
    assert brain.last_fire is None, "last_fire must be None when no fire occurred"
    traj2 = brain.get_trajectory()
    assert not any(pt.fired for pt in traj2), (
        "Task 2 trajectory must not show fired — stale state from task 1 would be a bug"
    )


def test_E_learned_motif_requires_exec_error():
    """LogicalFailureStore query in on_tool_call must not fire when execution
    succeeded (no exec_error), even if semantic similarity is high.

    Verifies: include_learned=False in before_tool_call, and the exec_error
    gate in on_tool_call's logic_warning computation.
    """
    brain = _make_brain()
    # before_tool_call: no learned signals, no surface motifs, no constraints
    brain._current_task_str = "Compute volatility."
    # Call before_tool_call with code that has no motif match
    code = "vol = np.std(returns) * 252**0.5"
    pre = brain.before_tool_call("python_exec", {"code": code})
    # No seeded motifs → should not fire
    assert pre is None, f"before_tool_call fired unexpectedly: {pre!r}"

    # on_tool_call with SUCCESS result (no error) — even high logic similarity must not fire
    success_result = "0.2134"   # plain number, no error
    post = brain.on_tool_call("python_exec", {"code": code}, success_result)
    assert post is None, (
        f"on_tool_call fired on successful execution — learned motif leaked: {post!r}"
    )
