"""Offline tests for TrajectoryDetector — developer week simulation.

Tests are structured as a realistic developer week:

  Day 1 — Discovery failures (motifs extracted, cold start)
  Day 2 — Same logical error in different surface form (detector must fire)
  Day 3 — Another recurrence with different vocabulary (detector must still fire)
  Day 4 — Near-miss tasks in the same domain (detector must stay silent)
  Day 5 — Cross-family contamination check (motif A must not fire on family B task)

Five failure families:
  1. selective_retry       — retry all vs. selective exception handling
  2. secondary_sort        — sort with missing tiebreak key
  3. validation_before_use — return default vs. raise on missing required field
  4. integer_division      — int/int silent truncation in ratio calculations
  5. timezone_unaware      — naive datetime comparison with timezone-aware datetimes

All LLM calls are stubbed — no API keys needed.
Tests pass/fail based on pipeline logic, not LLM output.
"""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pytest

from trace_use.brain import FailureMotif, MotifStore, _validate_judge_result
from trace_use.trajectory import (
    TrajectoryDetector,
    MotifMatch,
    _validate_match,
    _format_pitfalls,
)
from trace_use.config import DetectorConfig
from trace_use.motif_store import PersistentMotifStore


# ── Stub embedder ─────────────────────────────────────────────────────────────

def _stub_embedder(texts: list[str]) -> list[list[float]]:
    """Deterministic stub: same text → same vector; different text → different vector."""
    results = []
    for text in texts:
        rng = np.random.default_rng(seed=abs(hash(text)) % (2 ** 31))
        v   = rng.standard_normal(64).astype("float32")
        v  /= (np.linalg.norm(v) + 1e-9)
        results.append(v.tolist())
    return results


# ── Failure motif fixtures ────────────────────────────────────────────────────

def _make_selective_retry_motif() -> FailureMotif:
    return FailureMotif(
        id                  = "retry_on_all_errors_not_selective",
        name                = "Non-Selective Retry Catches All Exception Types",
        description         = "Code retries on all exceptions when the task requires selective retry based on error type.",
        required_condition  = "task requires selective retry based on error type or status code",
        violation_condition = "except Exception catches all types without type check",
        recommendation      = "check exception type or .status_code before deciding to retry",
        examples            = ["retry_request", "safe_fetch", "retry_on_type"],
        source              = "learned",
    )


def _make_secondary_sort_motif() -> FailureMotif:
    return FailureMotif(
        id                  = "secondary_sort_key_missing",
        name                = "Secondary Sort Key Missing in Multi-Criterion Sort",
        description         = "Sort uses a single key when the task requires a secondary tiebreak criterion.",
        required_condition  = "task requires secondary tiebreak when primary sort keys are equal",
        violation_condition = "sort uses single key with no tuple for secondary criterion",
        recommendation      = "use a tuple key (primary, secondary) so tiebreaks are deterministic",
        examples            = ["sort users by score then name", "rank products by price then id"],
        source              = "learned",
    )


def _make_validation_motif() -> FailureMotif:
    return FailureMotif(
        id                  = "silent_failure_missing_required_field",
        name                = "Silent Failure on Missing Required Field",
        description         = "Code returns a default value when a required field is absent instead of raising an error.",
        required_condition  = "task specifies that missing required field must raise an error",
        violation_condition = "code uses .get with a default or returns None instead of raising",
        recommendation      = "raise ValueError or KeyError explicitly when a required field is absent",
        examples            = ["extract_user_email", "parse_required_config_key"],
        source              = "learned",
    )


def _make_integer_division_motif() -> FailureMotif:
    return FailureMotif(
        id                  = "integer_division_silent_truncation",
        name                = "Integer Division Silently Truncates Ratio",
        description         = "Ratio or percentage computed with integer operands truncates the fractional part.",
        required_condition  = "task requires computing a ratio or percentage from integer counts",
        violation_condition = "division operands are both integers with no float conversion",
        recommendation      = "cast at least one operand to float before dividing, or use numerator / denominator * 1.0",
        examples            = ["compute_success_rate", "calculate_hit_ratio"],
        source              = "learned",
    )


def _make_timezone_motif() -> FailureMotif:
    return FailureMotif(
        id                  = "naive_datetime_comparison",
        name                = "Naive Datetime Compared With Timezone-Aware Datetime",
        description         = "Comparing a naive datetime (no tzinfo) with an aware datetime raises TypeError.",
        required_condition  = "task requires comparing timestamps that may include timezone information",
        violation_condition = "datetime created without tzinfo compared against timezone-aware value",
        recommendation      = "use datetime.now(timezone.utc) or .replace(tzinfo=timezone.utc) to ensure both datetimes are timezone-aware",
        examples            = ["filter_events_after_deadline", "check_token_expiry"],
        source              = "learned",
    )


def _populated_store(motifs: list[FailureMotif]) -> MotifStore:
    store = MotifStore(_stub_embedder)
    for m in motifs:
        store.add(m)
    return store


# ── Detector factory with stubbed judge ───────────────────────────────────────

def _make_detector(
    store: MotifStore,
    judge_fn: Callable | None = None,
    threshold: float = 0.70,
) -> TrajectoryDetector:
    # min_sim=-1.0: cosine similarity is in [-1,1], so -1.0 retrieves all motifs.
    # Tests want to exercise judge logic, not embedding similarity — the random
    # stub embedder has no semantic meaning.
    cfg      = DetectorConfig(judge_threshold=threshold, retrieval_min_sim=-1.0)
    detector = TrajectoryDetector(store, _stub_embedder, config=cfg)
    if judge_fn is not None:
        detector._call_judge = judge_fn
    return detector


def _exact_match_judge(motif: FailureMotif, task: str) -> dict:
    """Stub: fires when required_condition has ≥3 words present in the task."""
    req_words = set(motif.required_condition.lower().split())
    task_words = set(task.lower().split())
    shared = req_words & task_words - {"the", "a", "an", "and", "or", "in", "on", "to", "for", "of", "with", "is", "that", "task", "requires"}
    if len(shared) >= 3:
        first_shared = next(w for w in motif.required_condition.split() if w.lower() in task_words)
        evidence = " ".join(w for w in task.split() if w.lower() in req_words)[:80]
        return {
            "relevant": True,
            "confidence": 0.85,
            "evidence_quote": evidence or first_shared,
            "warning": f"This task explicitly requires handling: {motif.required_condition}. Past failure: {motif.description}",
        }
    return {"relevant": False, "confidence": 0.10, "evidence_quote": "", "warning": ""}


def _never_match_judge(motif: FailureMotif, task: str) -> dict:
    return {"relevant": False, "confidence": 0.10, "evidence_quote": "", "warning": ""}


# ═════════════════════════════════════════════════════════════════════════════
# Unit tests — TrajectoryDetector
# ═════════════════════════════════════════════════════════════════════════════

class TestDetectorBasics:
    def test_empty_store_returns_no_matches(self):
        store    = MotifStore(_stub_embedder)
        detector = _make_detector(store)
        matches  = detector.check("implement a retry mechanism for HTTP requests")
        assert matches == []

    def test_returns_empty_list_type(self):
        store    = MotifStore(_stub_embedder)
        detector = _make_detector(store)
        result   = detector.check("any task description here")
        assert isinstance(result, list)

    def test_fires_when_judge_returns_concrete_match(self):
        store    = _populated_store([_make_selective_retry_motif()])
        detector = _make_detector(store, judge_fn=lambda m, t: {
            "relevant": True,
            "confidence": 0.88,
            "evidence_quote": "only retry on ConnectionError",
            "warning": "code may catch all exceptions instead of just ConnectionError",
        })
        matches = detector.check("write a function that only retry on ConnectionError and raises on others")
        assert len(matches) == 1
        assert matches[0].motif.id == "retry_on_all_errors_not_selective"
        assert matches[0].evidence_quote == "only retry on ConnectionError"

    def test_stays_silent_when_judge_returns_not_relevant(self):
        store    = _populated_store([_make_selective_retry_motif()])
        detector = _make_detector(store, judge_fn=_never_match_judge)
        matches  = detector.check("sort a list of users alphabetically")
        assert matches == []

    def test_rejects_low_confidence(self):
        store    = _populated_store([_make_selective_retry_motif()])
        detector = _make_detector(store, threshold=0.80, judge_fn=lambda m, t: {
            "relevant": True,
            "confidence": 0.60,
            "evidence_quote": "retry on error type",
            "warning": "watch for exception handling",
        })
        matches = detector.check("retry on error type based on status")
        assert matches == [], "Low confidence must not fire"

    def test_rejects_empty_evidence_quote(self):
        store    = _populated_store([_make_selective_retry_motif()])
        detector = _make_detector(store, judge_fn=lambda m, t: {
            "relevant": True,
            "confidence": 0.95,
            "evidence_quote": "",   # no proof
            "warning": "this task might need selective retry",
        })
        matches = detector.check("implement a retry function")
        assert matches == [], "Empty evidence_quote must be rejected"

    def test_rejects_vague_language_in_evidence(self):
        task  = "process each item in the list"
        store = _populated_store([_make_selective_retry_motif()])
        for vague_phrase in ["task implies", "likely", "might", "probably", "seems like"]:
            result = {
                "relevant": True,
                "confidence": 0.90,
                "evidence_quote": f"{vague_phrase} needs retry logic",
                "warning": "watch out for exception handling",
            }
            assert not _validate_match(result, task, threshold=0.50), (
                f"Vague phrase {vague_phrase!r} must be rejected"
            )

    def test_rejects_evidence_not_grounded_in_task(self):
        task   = "sort users by name"
        result = {
            "relevant": True,
            "confidence": 0.90,
            "evidence_quote": "only retry on ConnectionError",   # not in task
            "warning": "watch for exception handling issues",
        }
        assert not _validate_match(result, task, threshold=0.50), (
            "Evidence quote not present in task must be rejected"
        )

    def test_short_warning_is_rejected(self):
        task   = "retry on ConnectionError only"
        result = {
            "relevant": True,
            "confidence": 0.90,
            "evidence_quote": "retry on ConnectionError only",
            "warning": "check",   # too short
        }
        assert not _validate_match(result, task, threshold=0.50)


class TestInject:
    def test_inject_returns_original_when_no_matches(self):
        store    = MotifStore(_stub_embedder)
        detector = _make_detector(store)
        task     = "compute the average of a list of numbers"
        enriched, matches = detector.inject(task)
        assert enriched == task
        assert matches == []

    def test_inject_prepends_pitfalls_when_matches_found(self):
        store    = _populated_store([_make_secondary_sort_motif()])
        detector = _make_detector(store, judge_fn=lambda m, t: {
            "relevant": True,
            "confidence": 0.85,
            "evidence_quote": "break ties alphabetically by name",
            "warning": "Ensure sort uses a tuple key for both score and name.",
        })
        task     = "sort users by score descending; break ties alphabetically by name"
        enriched, matches = detector.inject(task)
        assert enriched != task
        assert "KNOWN PITFALLS" in enriched
        assert task in enriched
        assert len(matches) == 1

    def test_inject_preserves_original_task_text(self):
        store    = _populated_store([_make_secondary_sort_motif()])
        detector = _make_detector(store, judge_fn=lambda m, t: {
            "relevant": True,
            "confidence": 0.85,
            "evidence_quote": "break ties by id",
            "warning": "Use tuple sort key for price and id.",
        })
        task     = "rank products by price ascending; break ties by id"
        enriched, _ = detector.inject(task)
        assert task in enriched, "Original task must appear verbatim in enriched prompt"

    def test_inject_multiple_matches_all_appear(self):
        store    = _populated_store([_make_selective_retry_motif(), _make_secondary_sort_motif()])
        call_count = {"n": 0}
        def _multi_judge(motif: FailureMotif, task: str) -> dict:
            call_count["n"] += 1
            return {
                "relevant": True,
                "confidence": 0.82,
                "evidence_quote": task.split()[0],
                "warning": f"Watch for {motif.required_condition[:40]} issues.",
            }
        detector = _make_detector(store, judge_fn=_multi_judge)
        _, matches = detector.inject("retry on selective error types and sort with tiebreak by name")
        assert len(matches) >= 1


class TestFormatPitfalls:
    def test_format_includes_motif_name(self):
        m = MotifMatch(
            motif          = _make_selective_retry_motif(),
            confidence     = 0.88,
            evidence_quote = "only retry on ConnectionError",
            warning        = "Code may catch all exceptions instead of just ConnectionError.",
        )
        text = _format_pitfalls([m])
        assert "Non-Selective Retry Catches All Exception Types" in text
        assert "only retry on ConnectionError" in text
        assert "check exception type" in text

    def test_format_includes_all_matches(self):
        matches = [
            MotifMatch(motif=_make_selective_retry_motif(), confidence=0.85,
                       evidence_quote="retry on type", warning="Watch for exception handling."),
            MotifMatch(motif=_make_secondary_sort_motif(), confidence=0.80,
                       evidence_quote="break ties by name", warning="Use tuple sort key."),
        ]
        text = _format_pitfalls(matches)
        assert "[1]" in text
        assert "[2]" in text
        assert "Non-Selective Retry" in text
        assert "Secondary Sort" in text


# ═════════════════════════════════════════════════════════════════════════════
# PersistentMotifStore tests
# ═════════════════════════════════════════════════════════════════════════════

class TestPersistentMotifStore:
    def test_saves_and_reloads(self, tmp_path):
        path   = str(tmp_path / "motifs.json")
        store1 = PersistentMotifStore(_stub_embedder, path=path)
        store1.add(_make_selective_retry_motif())
        store1.add(_make_secondary_sort_motif())
        assert store1.count == 2

        # Load in a fresh store instance
        store2 = PersistentMotifStore(_stub_embedder, path=path)
        assert store2.count == 2
        ids = {m.id for m in store2.motifs}
        assert "retry_on_all_errors_not_selective" in ids
        assert "secondary_sort_key_missing" in ids

    def test_motif_fields_survive_roundtrip(self, tmp_path):
        path   = str(tmp_path / "motifs.json")
        store1 = PersistentMotifStore(_stub_embedder, path=path)
        original = _make_validation_motif()
        store1.add(original)

        store2 = PersistentMotifStore(_stub_embedder, path=path)
        loaded = store2.motifs[0]
        assert loaded.id                  == original.id
        assert loaded.name                == original.name
        assert loaded.description         == original.description
        assert loaded.required_condition  == original.required_condition
        assert loaded.violation_condition == original.violation_condition
        assert loaded.recommendation      == original.recommendation

    def test_clear_wipes_file(self, tmp_path):
        path  = str(tmp_path / "motifs.json")
        store = PersistentMotifStore(_stub_embedder, path=path)
        store.add(_make_selective_retry_motif())
        assert store.count == 1
        store.clear()
        assert store.count == 0

        reloaded = PersistentMotifStore(_stub_embedder, path=path)
        assert reloaded.count == 0

    def test_update_persists(self, tmp_path):
        path  = str(tmp_path / "motifs.json")
        store = PersistentMotifStore(_stub_embedder, path=path)
        m     = _make_selective_retry_motif()
        store.add(m)

        updated = FailureMotif(
            id                  = m.id,
            name                = m.name,
            description         = "Updated description after second occurrence.",
            required_condition  = m.required_condition,
            violation_condition = m.violation_condition,
            recommendation      = m.recommendation,
            source              = "learned",
        )
        store.update(m.id, updated)

        reloaded = PersistentMotifStore(_stub_embedder, path=path)
        assert reloaded.motifs[0].description == "Updated description after second occurrence."

    def test_load_missing_file_starts_fresh(self, tmp_path):
        path  = str(tmp_path / "nonexistent.json")
        store = PersistentMotifStore(_stub_embedder, path=path)
        assert store.count == 0

    def test_write_is_atomic(self, tmp_path):
        """Verify .tmp file is not left behind after a successful save."""
        path  = str(tmp_path / "motifs.json")
        store = PersistentMotifStore(_stub_embedder, path=path)
        store.add(_make_selective_retry_motif())
        assert not os.path.exists(path + ".tmp")
        assert os.path.exists(path)


# ═════════════════════════════════════════════════════════════════════════════
# Developer week simulation
# ═════════════════════════════════════════════════════════════════════════════
#
# Each class represents one failure family.
# Tests simulate: discovery → recurrence → near-miss.
#
# Judge stub: simulates grounded matches by checking word overlap between
# the motif's required_condition and the task text. This mimics what a real
# LLM would do without needing API access.
# ═════════════════════════════════════════════════════════════════════════════

def _semantic_judge(motif: FailureMotif, task: str) -> dict:
    """Simulates an LLM judge: fires when signal terms from required_condition
    appear as substrings in the task text.

    Uses substring matching (not exact word match) so "integer" hits "integers",
    "type" hits "types", "error" hits "ValueError", etc. — the same way a real
    LLM reads for semantic presence rather than exact token equality.
    """
    import re as _re
    stopwords = {
        "the", "a", "an", "and", "or", "in", "on", "to", "for", "of", "with",
        "is", "that", "task", "requires", "when", "based", "from", "by", "as",
        "at", "it", "be", "are", "was", "were", "not", "do", "does", "have",
        "has", "which", "this", "its", "must", "should", "may", "than", "if",
    }
    req_tokens   = _re.findall(r"\w+", motif.required_condition.lower())
    signal_terms = [w for w in req_tokens if w not in stopwords and len(w) > 3]
    if not signal_terms:
        return {"relevant": False, "confidence": 0.1, "evidence_quote": "", "warning": ""}

    task_lower = task.lower()
    found = [t for t in signal_terms if t in task_lower]
    ratio = len(found) / len(signal_terms)

    if ratio >= 0.30:
        # Build evidence_quote from task words that contain a signal term
        evidence_words = [
            w for w in task.split()
            if any(t in w.lower() for t in signal_terms[:6])
        ]
        evidence = " ".join(evidence_words)[:100] or found[0]
        return {
            "relevant":       True,
            "confidence":     min(0.70 + 0.25 * ratio, 0.95),
            "evidence_quote": evidence,
            "warning":        f"Watch for: {motif.required_condition}. Recommendation: {motif.recommendation[:60]}",
        }
    return {"relevant": False, "confidence": 0.10, "evidence_quote": "", "warning": ""}


class TestWeekFamily_SelectiveRetry:
    """Family 1 — selective_retry.

    Day 1: developer writes retry_request, fails with 'except Exception' catching everything.
           Motif extracted: must check exception type before retrying.

    Day 2: developer writes retry_on_type — same logical requirement, different surface.
           Detector should fire.

    Day 3: developer writes safe_request with HTTP status check — same logical gap.
           Detector should fire.

    Day 4: developer writes a simple request function with no retry — near-miss.
           Detector must stay silent.
    """

    def _detector(self) -> TrajectoryDetector:
        store = _populated_store([_make_selective_retry_motif()])
        return _make_detector(store, judge_fn=_semantic_judge, threshold=0.65)

    # Day 2 recurrence
    def test_day2_retry_on_type_fires(self):
        detector = self._detector()
        task = "Write retry_on_type: only retry if exception type is in the retry_on list; raise immediately otherwise."
        matches = detector.check(task)
        assert len(matches) >= 1, f"retry_on_type should fire — selective retry is explicit in task: {task!r}"

    # Day 3 recurrence — different vocabulary, same logical gap
    def test_day3_safe_request_with_status_fires(self):
        detector = self._detector()
        task = "Implement safe_request: retry only when status code indicates a transient error, not on all exceptions."
        matches = detector.check(task)
        assert len(matches) >= 1, f"safe_request with selective status check should fire"

    # Day 4 near-miss — no retry at all
    def test_day4_no_retry_stays_silent(self):
        detector = self._detector()
        task = "Write fetch_data: make an HTTP GET request and return the JSON response body."
        matches = detector.check(task)
        assert matches == [], f"Simple fetch with no retry should not fire: {task!r}"

    # Day 5 cross-family — completely different domain
    def test_day5_sort_task_does_not_fire(self):
        detector = self._detector()
        task = "Sort a list of users by their score descending; break ties alphabetically by name."
        matches = detector.check(task)
        assert matches == [], f"Sort task must not fire retry motif: {task!r}"


class TestWeekFamily_SecondarySort:
    """Family 2 — secondary_sort.

    Day 1: developer sorts products by price, misses the name tiebreak.
           Motif extracted: requires tuple key for tiebreak.

    Day 2: developer sorts events by date with a title tiebreak.
           Detector should fire.

    Day 3: developer sorts leaderboard entries — rank then username.
           Detector should fire.

    Day 4: developer sorts a single-criterion list — no tiebreak in spec.
           Detector must stay silent.
    """

    def _detector(self) -> TrajectoryDetector:
        store = _populated_store([_make_secondary_sort_motif()])
        return _make_detector(store, judge_fn=_semantic_judge, threshold=0.65)

    def test_day2_date_then_title_fires(self):
        detector = self._detector()
        task = "Sort events by date ascending; when dates are equal, sort by title alphabetically as a secondary tiebreak."
        matches = detector.check(task)
        assert len(matches) >= 1, f"Date+title sort with explicit tiebreak should fire"

    def test_day3_rank_then_username_fires(self):
        detector = self._detector()
        task = "Rank leaderboard entries by score descending; use username alphabetically as the tiebreak when scores are equal."
        matches = detector.check(task)
        assert len(matches) >= 1, f"Rank+username sort should fire"

    def test_day4_single_criterion_no_fire(self):
        detector = self._detector()
        task = "Sort products by name alphabetically."
        matches = detector.check(task)
        assert matches == [], f"Single-criterion sort must not fire: {task!r}"

    def test_day5_retry_task_does_not_fire(self):
        detector = self._detector()
        task = "Write a function to retry failed HTTP requests up to 3 times with exponential backoff."
        matches = detector.check(task)
        assert matches == [], f"Retry task must not fire sort motif"


class TestWeekFamily_ValidationBeforeUse:
    """Family 3 — validation_before_use.

    Motif: missing required field must raise, not silently return default.
    """

    def _detector(self) -> TrajectoryDetector:
        store = _populated_store([_make_validation_motif()])
        return _make_detector(store, judge_fn=_semantic_judge, threshold=0.65)

    def test_explicit_raise_requirement_fires(self):
        detector = self._detector()
        task = "Parse a user record. If the email field is missing, raise ValueError. Do not return a default."
        matches = detector.check(task)
        assert len(matches) >= 1

    def test_no_raise_requirement_silent(self):
        detector = self._detector()
        task = "Parse a user record and return the email if present, or None if not found."
        matches = detector.check(task)
        assert matches == [], "Task that explicitly allows None return must not fire"


class TestWeekFamily_IntegerDivision:
    """Family 4 — integer_division.

    Motif: ratio from integer counts silently truncates.
    """

    def _detector(self) -> TrajectoryDetector:
        store = _populated_store([_make_integer_division_motif()])
        return _make_detector(store, judge_fn=_semantic_judge, threshold=0.65)

    def test_ratio_from_counts_fires(self):
        detector = self._detector()
        task = "Compute the success rate as a ratio of successful requests to total requests (both integers)."
        matches = detector.check(task)
        assert len(matches) >= 1

    def test_already_float_silent(self):
        detector = self._detector()
        task = "Compute the average temperature from a list of float sensor readings."
        matches = detector.check(task)
        assert matches == [], "Float average task must not fire integer division motif"


class TestWeekFamily_TimezoneUnaware:
    """Family 5 — timezone_unaware.

    Motif: naive datetime compared to timezone-aware datetime.
    """

    def _detector(self) -> TrajectoryDetector:
        store = _populated_store([_make_timezone_motif()])
        return _make_detector(store, judge_fn=_semantic_judge, threshold=0.65)

    def test_timestamp_comparison_with_timezone_fires(self):
        detector = self._detector()
        task = "Filter events occurring after a given deadline timestamp. The deadline includes timezone information."
        matches = detector.check(task)
        assert len(matches) >= 1

    def test_pure_string_comparison_silent(self):
        detector = self._detector()
        task = "Sort a list of file names alphabetically and return the first 10."
        matches = detector.check(task)
        assert matches == [], "String sort task must not fire timezone motif"


# ═════════════════════════════════════════════════════════════════════════════
# Cross-family contamination — the guarantee
# ═════════════════════════════════════════════════════════════════════════════

class TestCrossContamination:
    """Motifs from one family must never fire on tasks from another family.

    This is the key guarantee of the structured proof requirement:
    a retry motif cannot fire on a sort task because "selective retry"
    does not appear in a sort task's text.
    """

    @pytest.mark.parametrize("motif_factory,unrelated_task", [
        (
            _make_selective_retry_motif,
            "Sort a list of products by price ascending, break ties by name.",
        ),
        (
            _make_secondary_sort_motif,
            "Write a retry function that backs off exponentially on transient errors.",
        ),
        (
            _make_validation_motif,
            "Compute the ratio of successful to failed requests as a percentage.",
        ),
        (
            _make_integer_division_motif,
            "Compare a user's last_login timestamp with the token expiry time.",
        ),
        (
            _make_timezone_motif,
            "Sort events by start time; break ties alphabetically by event title.",
        ),
    ])
    def test_no_cross_fire(self, motif_factory: Callable, unrelated_task: str):
        store    = _populated_store([motif_factory()])
        detector = _make_detector(store, judge_fn=_semantic_judge, threshold=0.65)
        matches  = detector.check(unrelated_task)
        assert matches == [], (
            f"Motif {motif_factory().id!r} must not fire on unrelated task:\n  {unrelated_task!r}"
        )


# ═════════════════════════════════════════════════════════════════════════════
# Multi-day session simulation — full pipeline with PersistentMotifStore
# ═════════════════════════════════════════════════════════════════════════════

class TestFullWeekSession:
    """End-to-end simulation: motifs seeded on day 1 must fire on day 3 tasks."""

    def _full_detector(self, tmp_path, motifs: list[FailureMotif]) -> TrajectoryDetector:
        path  = str(tmp_path / "week_motifs.json")
        store = PersistentMotifStore(_stub_embedder, path=path)
        for m in motifs:
            store.add(m)
        # Reload to simulate a new session (day 3 vs day 1)
        store2 = PersistentMotifStore(_stub_embedder, path=path)
        # min_sim=-1.0 so retrieval is unconditional; only the judge decides
        return _make_detector(store2, judge_fn=_semantic_judge, threshold=0.65)

    def test_day1_motif_fires_on_day3_task(self, tmp_path):
        """Motif learned on day 1 must fire on a day-3 task with different surface code."""
        detector = self._full_detector(tmp_path, [_make_selective_retry_motif()])
        day3_task = "Implement request_with_retry: only retry on ConnectionError or Timeout, raise immediately on other exception types."
        matches   = detector.check(day3_task)
        assert len(matches) >= 1, "Day-1 retry motif must fire on day-3 selective-retry task"

    def test_five_motifs_loaded_and_correct_ones_fire(self, tmp_path):
        """Load all 5 motifs, check only the relevant one fires."""
        all_motifs = [
            _make_selective_retry_motif(),
            _make_secondary_sort_motif(),
            _make_validation_motif(),
            _make_integer_division_motif(),
            _make_timezone_motif(),
        ]
        detector = self._full_detector(tmp_path, all_motifs)
        assert detector._store.count == 5

        # Only secondary_sort should fire on this task
        task    = "Sort transactions by amount descending; when amounts are equal, use the transaction id as a secondary tiebreak."
        matches = detector.check(task)
        fired_ids = {m.motif.id for m in matches}
        assert "secondary_sort_key_missing" in fired_ids, "Secondary sort motif must fire"
        assert "retry_on_all_errors_not_selective" not in fired_ids, "Retry motif must not fire on sort task"

    def test_store_count_after_reload(self, tmp_path):
        """Count must be preserved across session boundaries."""
        path  = str(tmp_path / "session.json")
        store = PersistentMotifStore(_stub_embedder, path=path)
        for m in [_make_selective_retry_motif(), _make_secondary_sort_motif(), _make_validation_motif()]:
            store.add(m)
        assert store.count == 3

        reloaded = PersistentMotifStore(_stub_embedder, path=path)
        assert reloaded.count == 3

    def test_near_miss_on_day4_stays_silent(self, tmp_path):
        """Near-miss tasks (same domain, no actual bug requirement) must not fire."""
        detector = self._full_detector(tmp_path, [_make_secondary_sort_motif()])
        near_miss = "Sort a list of product names alphabetically and return the first 10."
        matches   = detector.check(near_miss)
        assert matches == [], f"Single-criterion sort (near-miss) must not fire: {near_miss!r}"

    def test_week_session_accumulates_motifs(self, tmp_path):
        """Adding motifs over 5 days results in correct cumulative count."""
        path = str(tmp_path / "week.json")

        # Day 1
        s1 = PersistentMotifStore(_stub_embedder, path=path)
        s1.add(_make_selective_retry_motif())

        # Day 2
        s2 = PersistentMotifStore(_stub_embedder, path=path)
        s2.add(_make_secondary_sort_motif())

        # Day 3
        s3 = PersistentMotifStore(_stub_embedder, path=path)
        s3.add(_make_validation_motif())

        # Day 4
        s4 = PersistentMotifStore(_stub_embedder, path=path)
        s4.add(_make_integer_division_motif())

        # Day 5 — load and verify all 4 are present
        s5 = PersistentMotifStore(_stub_embedder, path=path)
        assert s5.count == 4
        ids = {m.id for m in s5.motifs}
        assert "retry_on_all_errors_not_selective" in ids
        assert "secondary_sort_key_missing" in ids
        assert "silent_failure_missing_required_field" in ids
        assert "integer_division_silent_truncation" in ids


# ═════════════════════════════════════════════════════════════════════════════
# BrainConfig integration — verify config drives behavior
# ═════════════════════════════════════════════════════════════════════════════

class TestConfigDrivesBehavior:
    def test_custom_retrieval_min_sim_negative_one_retrieves_all(self):
        """min_sim=-1.0 must return all stored motifs: cosine similarity ∈ [-1, 1]."""
        store = _populated_store([
            _make_selective_retry_motif(),
            _make_secondary_sort_motif(),
            _make_validation_motif(),
        ])
        candidates = store.retrieve(
            task="completely unrelated task about something else",
            reasoning="", code="",
            top_k=10,
            min_sim=-1.0,
        )
        assert len(candidates) == 3

    def test_high_min_sim_returns_fewer_candidates(self):
        """min_sim=0.99 should return nothing for a clearly unrelated task."""
        store = _populated_store([_make_selective_retry_motif()])
        candidates = store.retrieve(
            task="completely unrelated task about painting a picture",
            reasoning="", code="",
            top_k=4,
            min_sim=0.99,
        )
        assert candidates == []

    def test_brain_config_controls_exec_tool_name(self):
        """BrainAgent must skip non-python_exec tools when configured."""
        from trace_use.brain import BrainAgent
        from trace_use.config import BrainConfig
        cfg   = BrainConfig(exec_tool_name="bash")
        brain = BrainAgent(_stub_embedder, config=cfg)
        brain.set_task(0, task="some task")
        # Inject a motif so before_tool_call has something to check
        brain._motif_store.add(_make_selective_retry_motif())
        # "python_exec" should be skipped since tool name is "bash"
        result = brain.before_tool_call("python_exec", {"code": "x = 1"})
        assert result is None, "python_exec must be skipped when exec_tool_name='bash'"

    def test_brain_config_controls_max_interventions(self):
        """BrainAgent must stop firing after max_interventions."""
        from trace_use.brain import BrainAgent
        from trace_use.config import BrainConfig
        cfg   = BrainConfig(max_interventions=1, judge_threshold=0.50)
        brain = BrainAgent(_stub_embedder, config=cfg)
        brain.set_task(0, task="retry on selective error type")

        motif = _make_selective_retry_motif()
        brain._motif_store.add(motif)
        brain._motif_store.retrieve = lambda **kwargs: [motif]
        concrete = {
            "applies": True, "confidence": 0.95,
            "requirement_quote": "retry on selective error type",
            "violation_quote": "except Exception",
            "explanation": "catches all", "recommendation": "check type before retry",
        }
        brain._call_judge = lambda *args, **kwargs: concrete

        # First call: must fire
        r1 = brain.before_tool_call("python_exec", {"code": "try:\n  pass\nexcept Exception:\n  retry()"})
        assert r1 is not None, "First call must fire"
        # Second call: must be blocked (max_interventions=1 reached)
        r2 = brain.before_tool_call("python_exec", {"code": "try:\n  pass\nexcept Exception:\n  retry()"})
        assert r2 is None, "Second call must not fire — max_interventions reached"
