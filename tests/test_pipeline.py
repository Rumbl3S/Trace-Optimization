"""Offline gates for the public API — no API key needed."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pipeline
from pipeline import Forecaster, decompose, attempt, gold_judge, tiered_judge, make_retriever


def test_decompose_caps_and_filters():
    agent = lambda p: ("What is the capital of France?\nWho leads it?\nok", 0)
    qs = decompose("task", agent, cap=2)
    assert qs == ["What is the capital of France?", "Who leads it?"]   # 'ok' too short, cap=2


def test_attempt_uses_context_and_question():
    seen = {}
    def agent(p):
        seen["p"] = p
        return ("reasoning... ANSWER: 42", 0)
    out = attempt("how many?", "the count is 42", agent)
    assert "42" in out and "the count is 42" in seen["p"] and "how many?" in seen["p"]


def test_gold_judge_pluggable_verifier():
    judge = gold_judge("Paris", lambda p: ("YES", 0))
    assert judge("capital?", "Paris") == 1.0
    judge_no = gold_judge("Paris", lambda p: ("NO", 0))
    assert judge_no("capital?", "London") == 0.0


def test_self_judge_needs_no_gold():
    assert pipeline.self_judge(lambda p: ("YES", 0))("q", "a") == 1.0
    assert pipeline.self_judge(lambda p: ("NO", 0))("q", "a") == 0.0


def test_self_judge_uses_evidence_when_given():
    seen = {}
    def judge(p):
        seen["p"] = p
        return ("YES", 0)
    pipeline.self_judge(judge, evidence_fn=lambda q: "the sky is blue")("q?", "blue")
    assert "the sky is blue" in seen["p"]


def test_self_consistency_no_labels_no_judge():
    consistent = pipeline.self_consistency(lambda q: "ANSWER: 42", samples=3)
    assert consistent("q", "ANSWER: 42") == 1.0          # all resamples agree -> confident
    assert consistent("q", "ANSWER: 99") == 0.0          # disagrees with the stable answer
    pool = iter(["ANSWER: 1", "ANSWER: 2", "ANSWER: 1"])
    flaky = pipeline.self_consistency(lambda q: next(pool), samples=3)
    assert 0.0 < flaky("q", "ANSWER: 1") < 1.0           # partial agreement -> middling


def _embed(texts):
    # deterministic stub: 'bad' traces -> one cluster, others -> the opposite
    return np.array([[1.0, 0.0] if "bad" in t else [0.0, 1.0] for t in texts])


def test_forecaster_predicts_and_gates():
    fc = Forecaster(_embed, k=1).fit(
        ["bad run", "bad attempt", "good run", "good attempt"], [0, 0, 1, 1])
    assert fc.predict_fail("a bad one") > 0.5
    assert fc.predict_fail("a good one") < 0.5
    assert fc.should_intervene("a bad one") and not fc.should_intervene("a good one")


def test_forecaster_add_grows_store():
    fc = Forecaster(_embed, k=1).fit(["good a"], [1])
    fc.add("bad b", 0)
    assert len(fc._labels) == 2 and fc.predict_fail("bad c") > 0.5


# ── tiered_judge ─────────────────────────────────────────────────────────────

def test_tiered_judge_fast_confident_pass_skips_strong():
    strong_calls = []
    def strong(p):
        strong_calls.append(p)
        return ("YES", 0)
    v = tiered_judge(fast_agent=lambda p: ("YES", 0), strong_agent=strong, gold="Paris")
    assert v("capital?", "Paris") == 1.0
    assert len(strong_calls) == 0   # haiku was confident; opus never called


def test_tiered_judge_fast_confident_fail_skips_strong():
    strong_calls = []
    v = tiered_judge(
        fast_agent=lambda p: ("NO", 0),
        strong_agent=lambda p: strong_calls.append(p) or ("YES", 0),
        gold="Paris",
    )
    assert v("capital?", "London") == 0.0
    assert len(strong_calls) == 0


def test_tiered_judge_escalates_when_uncertain():
    # gold_judge maps YES->1.0, NO->0.0 — to land in the band we need a score between 0.35-0.65.
    # Since gold_judge only returns 0.0 or 1.0, we test by narrowing the band to include 0.0.
    strong_calls = []
    def strong(p):
        strong_calls.append(p)
        return ("YES", 0)
    # band includes 0.0 -> haiku returns 0.0 -> falls in [0.0, 0.65] -> escalate
    v = tiered_judge(
        fast_agent=lambda p: ("NO", 0),
        strong_agent=strong,
        gold="Paris",
        uncertainty_band=(0.0, 0.65),
    )
    result = v("capital?", "Paris")
    assert len(strong_calls) == 1        # opus was called
    assert result == 1.0                 # opus said YES


def test_tiered_judge_returns_fast_verdict_outside_band():
    strong_calls = []
    v = tiered_judge(
        fast_agent=lambda p: ("YES", 0),
        strong_agent=lambda p: strong_calls.append(p) or ("NO", 0),
        gold="Paris",
        uncertainty_band=(0.4, 0.6),   # 1.0 is outside band
    )
    assert v("capital?", "Paris") == 1.0
    assert len(strong_calls) == 0


def test_tiered_judge_custom_band_boundaries():
    # score of 1.0: band (1.0, 1.0) — exactly on boundary, should escalate
    strong_calls = []
    v = tiered_judge(
        fast_agent=lambda p: ("YES", 0),
        strong_agent=lambda p: strong_calls.append(p) or ("NO", 0),
        gold="Paris",
        uncertainty_band=(1.0, 1.0),
    )
    v("q", "a")
    assert len(strong_calls) == 1


# ── make_retriever ────────────────────────────────────────────────────────────

def _unit_embedder(texts):
    """Maps each text to a 2-d vector by hashing first char, for deterministic retrieval."""
    vecs = []
    for t in texts:
        h = hash(t[0]) % 100 / 100.0 if t else 0.0
        vecs.append([h, 1.0 - h])
    v = np.array(vecs, dtype="float32")
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    return v / (norms + 1e-9)


def test_make_retriever_returns_string():
    retrieve = make_retriever(["alpha chunk", "beta chunk"], _unit_embedder)
    result = retrieve("alpha chunk")
    assert isinstance(result, str) and len(result) > 0


def test_make_retriever_empty_corpus():
    retrieve = make_retriever([], _unit_embedder)
    assert retrieve("anything") == ""


def test_make_retriever_whitespace_only_chunks_ignored():
    retrieve = make_retriever(["   ", "\n", "real content here"], _unit_embedder)
    result = retrieve("real content here")
    assert "real content here" in result


def test_make_retriever_respects_word_budget():
    # each chunk is ~10 words; budget of 12 should admit at most 1 full chunk
    chunks = ["word " * 10, "other " * 10, "third " * 10]
    retrieve = make_retriever(chunks, _unit_embedder)
    result = retrieve("word", words=12)
    assert len(result.split()) <= 15   # some slack for joining


def test_make_retriever_joins_multiple_chunks():
    chunks = ["aaa aaa", "bbb bbb", "ccc ccc"]
    retrieve = make_retriever(chunks, _unit_embedder)
    result = retrieve("aaa", words=100)
    assert "\n\n" in result   # multiple chunks joined


# ── decompose edge cases ──────────────────────────────────────────────────────

def test_decompose_fallback_when_all_lines_too_short():
    agent = lambda p: ("ok\nhi\nno", 0)   # all < 6 chars
    qs = decompose("my full task", agent)
    assert qs == ["my full task"]


def test_decompose_strips_bullets_and_dashes():
    agent = lambda p: ("- First sub-question here\n* Second sub-question here", 0)
    qs = decompose("task", agent)
    assert all(not q.startswith(("-", "*")) for q in qs)


def test_decompose_respects_cap():
    lines = "\n".join(f"sub-question number {i} here" for i in range(20))
    agent = lambda p: (lines, 0)
    assert len(decompose("task", agent, cap=3)) == 3


# ── self_consistency edge cases ───────────────────────────────────────────────

def test_self_consistency_samples_1():
    v = pipeline.self_consistency(lambda q: "ANSWER: 42", samples=1)
    assert v("q", "ANSWER: 42") == 1.0
    assert v("q", "ANSWER: 99") == 0.0


def test_self_consistency_samples_zero_treated_as_one():
    v = pipeline.self_consistency(lambda q: "ANSWER: 42", samples=0)
    assert v("q", "ANSWER: 42") == 1.0   # max(1, 0) == 1


# ── gold_judge robustness ─────────────────────────────────────────────────────

def test_gold_judge_case_insensitive_yes():
    for response in ("YES", "Yes", "yes", "YES extra text"):
        j = gold_judge("gold", lambda p: (response, 0))
        assert j("q", "a") == 1.0, f"failed on: {response!r}"


def test_gold_judge_no_when_not_yes():
    for response in ("NO", "no", "Not correct", "maybe"):
        j = gold_judge("gold", lambda p: (response, 0))
        assert j("q", "a") == 0.0, f"expected 0.0 for: {response!r}"


def test_gold_judge_truncates_long_answer():
    seen = {}
    def agent(p):
        seen["p"] = p
        return ("YES", 0)
    gold_judge("gold", agent)("q", "x" * 5000)
    assert len(seen["p"]) < 8000   # long answer was truncated before being sent


# ── Forecaster edge cases ─────────────────────────────────────────────────────

def test_forecaster_single_class_all_success_returns_low_fail():
    fc = Forecaster(_embed, k=3).fit(["good a", "good b", "good c"], [1, 1, 1])
    # single class -> fallback: base rate = 1 -> p_fail = 0
    assert fc.predict_fail("good d") == 0.0


def test_forecaster_single_class_all_failure_returns_high_fail():
    fc = Forecaster(_embed, k=3).fit(["bad a", "bad b", "bad c"], [0, 0, 0])
    assert fc.predict_fail("bad d") == 1.0


def test_forecaster_predict_fail_bounded():
    fc = Forecaster(_embed, k=2).fit(
        ["bad x", "bad y", "good x", "good y"], [0, 0, 1, 1])
    for trace in ["bad z", "good z", "neutral"]:
        p = fc.predict_fail(trace)
        assert 0.0 <= p <= 1.0, f"out of bounds for {trace!r}: {p}"


def test_forecaster_should_intervene_threshold():
    fc = Forecaster(_embed, k=1).fit(["bad run", "good run"], [0, 1])
    # default threshold 0.5
    assert fc.should_intervene("bad trace", threshold=0.5)
    assert not fc.should_intervene("good trace", threshold=0.5)
    # raising threshold should suppress intervention even on bad trace
    assert not fc.should_intervene("bad trace", threshold=1.1)
