"""Offline gates for the public API — no API key needed.

Coverage:
  decompose / attempt / gold_judge / self_judge / self_consistency
  tiered_judge / code_judge / extract_code / make_retriever
  Forecaster (kNN + PCA) / TaskResult / ComponentResult / run_task
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pipeline
from pipeline import (
    Forecaster, TaskResult, ComponentResult,
    decompose, attempt,
    gold_judge, self_judge, self_consistency, tiered_judge,
    code_judge, extract_code,
    make_retriever,
)


# ── shared stubs ──────────────────────────────────────────────────────────────

def _embed(texts):
    """2-d stub: 'bad' -> [1,0] cluster, anything else -> [0,1] cluster."""
    return np.array([[1.0, 0.0] if "bad" in t else [0.0, 1.0] for t in texts])


def _embed3d(texts):
    """3-d stub for PCA tests: 'bad' -> [1,0,0], 'good' -> [0,1,0], else [0,0,1]."""
    def _v(t):
        if "bad"  in t: return [1.0, 0.0, 0.0]
        if "good" in t: return [0.0, 1.0, 0.0]
        return [0.0, 0.0, 1.0]
    return np.array([_v(t) for t in texts])


def _yes_agent(p): return ("YES", 0)
def _no_agent(p):  return ("NO",  0)


# ═══════════════════════════════════════════════════════════════════════════════
# decompose
# ═══════════════════════════════════════════════════════════════════════════════

def test_decompose_caps_and_filters():
    agent = lambda p: ("What is the capital of France?\nWho leads it?\nok", 0)
    qs = decompose("task", agent, cap=2)
    assert qs == ["What is the capital of France?", "Who leads it?"]


def test_decompose_fallback_when_all_lines_too_short():
    agent = lambda p: ("ok\nhi\nno", 0)
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


def test_decompose_default_cap_is_8():
    lines = "\n".join(f"sub-question number {i} here" for i in range(20))
    agent = lambda p: (lines, 0)
    assert len(decompose("task", agent)) == 8


def test_decompose_strips_tabs_and_stars():
    agent = lambda p: ("\t* question alpha here\n\t- question beta here", 0)
    qs = decompose("task", agent)
    assert all(q[0] not in "*-\t" for q in qs)


def test_decompose_single_line_output():
    agent = lambda p: ("What is the meaning of life here?", 0)
    qs = decompose("task", agent)
    assert len(qs) == 1 and "meaning of life" in qs[0]


def test_decompose_accepts_plain_text_response():
    agent = lambda p: ("Question one is this one\nQuestion two is that one", 0)
    qs = decompose("task", agent)
    assert len(qs) == 2


# ═══════════════════════════════════════════════════════════════════════════════
# attempt
# ═══════════════════════════════════════════════════════════════════════════════

def test_attempt_uses_context_and_question():
    seen = {}
    def agent(p):
        seen["p"] = p
        return ("reasoning... ANSWER: 42", 0)
    out = attempt("how many?", "the count is 42", agent)
    assert "42" in out and "the count is 42" in seen["p"] and "how many?" in seen["p"]


def test_attempt_empty_context():
    agent = lambda p: ("ANSWER: yes", 0)
    out = attempt("will it work?", "", agent)
    assert "yes" in out.lower()


def test_attempt_returns_string():
    agent = lambda p: ("some trace", 0)
    assert isinstance(attempt("q", "ctx", agent), str)


def test_attempt_agent_returning_plain_string():
    agent = lambda p: "plain text"
    assert attempt("q", "ctx", agent) == "plain text"


# ═══════════════════════════════════════════════════════════════════════════════
# gold_judge
# ═══════════════════════════════════════════════════════════════════════════════

def test_gold_judge_pluggable_verifier():
    assert gold_judge("Paris", _yes_agent)("capital?", "Paris") == 1.0
    assert gold_judge("Paris", _no_agent)("capital?", "London") == 0.0


def test_gold_judge_case_insensitive_yes():
    for response in ("YES", "Yes", "yes", "YES extra text"):
        j = gold_judge("gold", lambda p: (response, 0))
        assert j("q", "a") == 1.0, f"failed on: {response!r}"


def test_gold_judge_no_when_not_yes():
    for response in ("NO", "no", "Not correct", "maybe", ""):
        j = gold_judge("gold", lambda p: (response, 0))
        assert j("q", "a") == 0.0, f"expected 0.0 for: {response!r}"


def test_gold_judge_truncates_long_answer():
    seen = {}
    def agent(p):
        seen["p"] = p
        return ("YES", 0)
    gold_judge("gold", agent)("q", "x" * 5000)
    assert len(seen["p"]) < 10000


def test_gold_judge_includes_gold_in_prompt():
    seen = {}
    def agent(p):
        seen["p"] = p
        return ("YES", 0)
    gold_judge("the-gold-value", agent)("q", "a")
    assert "the-gold-value" in seen["p"]


def test_gold_judge_includes_question_in_prompt():
    seen = {}
    def agent(p):
        seen["p"] = p
        return ("YES", 0)
    gold_judge("gold", agent)("what-is-the-question?", "a")
    assert "what-is-the-question?" in seen["p"]


# ═══════════════════════════════════════════════════════════════════════════════
# self_judge
# ═══════════════════════════════════════════════════════════════════════════════

def test_self_judge_needs_no_gold():
    assert self_judge(_yes_agent)("q", "a") == 1.0
    assert self_judge(_no_agent)("q", "a") == 0.0


def test_self_judge_uses_evidence_when_given():
    seen = {}
    def judge(p):
        seen["p"] = p
        return ("YES", 0)
    self_judge(judge, evidence_fn=lambda q: "the sky is blue")("q?", "blue")
    assert "the sky is blue" in seen["p"]


def test_self_judge_sees_full_trace_not_just_tail():
    seen = {}
    def judge(p):
        seen["p"] = p
        return ("YES", 0)
    # an 8k trace is under the 16k cap, so the WHOLE trace must reach the judge —
    # the executed evidence (head) and the conclusion (tail) are both needed.
    head = "HEADTOKEN " * 400   # 4000 chars
    tail = "TAILTOKEN " * 400   # 4000 chars
    self_judge(judge)("q", head + tail)
    assert "TAILTOKEN" in seen["p"]
    assert "HEADTOKEN" in seen["p"]


def test_self_judge_caps_pathologically_long_answer():
    seen = {}
    def judge(p):
        seen["p"] = p
        return ("YES", 0)
    # beyond 16k the front is dropped as a runaway guard; the tail (conclusion +
    # most recent evidence) is what's kept.
    answer = "FRONTMARKER " + ("x" * 20000) + " ENDMARKER"
    self_judge(judge)("q", answer)
    assert "ENDMARKER" in seen["p"]
    assert "FRONTMARKER" not in seen["p"]


def test_self_judge_evidence_fn_receives_question():
    received = {}
    def evidence_fn(q):
        received["q"] = q
        return "some evidence"
    self_judge(_yes_agent, evidence_fn=evidence_fn)("the-question", "a")
    assert received["q"] == "the-question"


def test_self_judge_no_evidence_fn_skips_evidence():
    seen = {}
    def judge(p):
        seen["p"] = p
        return ("YES", 0)
    self_judge(judge)("q", "a")
    assert "Evidence:" not in seen["p"]


# ═══════════════════════════════════════════════════════════════════════════════
# self_consistency
# ═══════════════════════════════════════════════════════════════════════════════

def test_self_consistency_no_labels_no_judge():
    consistent = pipeline.self_consistency(lambda q: "ANSWER: 42", samples=3)
    assert consistent("q", "ANSWER: 42") == 1.0
    assert consistent("q", "ANSWER: 99") == 0.0
    pool = iter(["ANSWER: 1", "ANSWER: 2", "ANSWER: 1"])
    flaky = pipeline.self_consistency(lambda q: next(pool), samples=3)
    assert 0.0 < flaky("q", "ANSWER: 1") < 1.0


def test_self_consistency_samples_1():
    v = pipeline.self_consistency(lambda q: "ANSWER: 42", samples=1)
    assert v("q", "ANSWER: 42") == 1.0
    assert v("q", "ANSWER: 99") == 0.0


def test_self_consistency_samples_zero_treated_as_one():
    v = pipeline.self_consistency(lambda q: "ANSWER: 42", samples=0)
    assert v("q", "ANSWER: 42") == 1.0


def test_self_consistency_high_samples():
    v = pipeline.self_consistency(lambda q: "ANSWER: 7", samples=10)
    assert v("q", "ANSWER: 7") == 1.0


def test_self_consistency_returns_fraction():
    pool = iter(["ANSWER: A", "ANSWER: B", "ANSWER: A", "ANSWER: A"])
    v = pipeline.self_consistency(lambda q: next(pool), samples=4)
    score = v("q", "ANSWER: A")
    assert abs(score - 3/4) < 1e-9


def test_self_consistency_distinct_answers_disagree():
    # resample always returns "ANSWER: foo", but target is "ANSWER: bar" -> 0.0
    v = pipeline.self_consistency(lambda q: "ANSWER: foo", samples=3)
    assert v("q", "ANSWER: bar") == 0.0
    assert v("q", "ANSWER: foo") == 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# tiered_judge
# ═══════════════════════════════════════════════════════════════════════════════

def test_tiered_judge_fast_confident_pass_skips_strong():
    strong_calls = []
    v = tiered_judge(fast_agent=_yes_agent,
                     strong_agent=lambda p: strong_calls.append(p) or ("YES", 0),
                     gold="Paris")
    assert v("capital?", "Paris") == 1.0
    assert len(strong_calls) == 0


def test_tiered_judge_fast_confident_fail_skips_strong():
    strong_calls = []
    v = tiered_judge(fast_agent=_no_agent,
                     strong_agent=lambda p: strong_calls.append(p) or ("YES", 0),
                     gold="Paris")
    assert v("capital?", "London") == 0.0
    assert len(strong_calls) == 0


def test_tiered_judge_escalates_when_uncertain():
    strong_calls = []
    def strong(p):
        strong_calls.append(p)
        return ("YES", 0)
    v = tiered_judge(fast_agent=_no_agent, strong_agent=strong,
                     gold="Paris", uncertainty_band=(0.0, 0.65))
    result = v("capital?", "Paris")
    assert len(strong_calls) == 1
    assert result == 1.0


def test_tiered_judge_returns_fast_verdict_outside_band():
    strong_calls = []
    v = tiered_judge(fast_agent=_yes_agent,
                     strong_agent=lambda p: strong_calls.append(p) or ("NO", 0),
                     gold="Paris", uncertainty_band=(0.4, 0.6))
    assert v("capital?", "Paris") == 1.0
    assert len(strong_calls) == 0


def test_tiered_judge_custom_band_boundaries():
    strong_calls = []
    v = tiered_judge(fast_agent=_yes_agent,
                     strong_agent=lambda p: strong_calls.append(p) or ("NO", 0),
                     gold="Paris", uncertainty_band=(1.0, 1.0))
    v("q", "a")
    assert len(strong_calls) == 1


def test_tiered_judge_strong_verdict_returned_on_escalation():
    v = tiered_judge(fast_agent=_no_agent, strong_agent=_yes_agent,
                     gold="Paris", uncertainty_band=(0.0, 0.65))
    assert v("q", "a") == 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# extract_code
# ═══════════════════════════════════════════════════════════════════════════════

def test_extract_code_python_fenced():
    text = "Sure!\n```python\ndef foo():\n    return 42\n```\n"
    assert "def foo" in extract_code(text)
    assert "```" not in extract_code(text)


def test_extract_code_py_fenced():
    text = "```py\nx = 1\n```"
    assert extract_code(text) == "x = 1"


def test_extract_code_generic_fenced():
    text = "```\nx = 2\n```"
    assert extract_code(text) == "x = 2"


def test_extract_code_no_fence_returns_text():
    text = "x = 3"
    assert extract_code(text) == "x = 3"


def test_extract_code_takes_first_block_only():
    text = "```python\nfirst = 1\n```\n```python\nsecond = 2\n```"
    code = extract_code(text)
    assert "first" in code and "second" not in code


def test_extract_code_empty_string():
    assert extract_code("") == ""


def test_extract_code_strips_whitespace():
    text = "```python\n   def f(): pass   \n```"
    assert extract_code(text) == "def f(): pass"


def test_extract_code_case_insensitive_lang_tag():
    text = "```Python\ndef bar(): pass\n```"
    assert "def bar" in extract_code(text)


# ═══════════════════════════════════════════════════════════════════════════════
# code_judge
# ═══════════════════════════════════════════════════════════════════════════════

def test_code_judge_pass_when_check_true():
    answer = "```python\ndef add(a, b): return a + b\n```"
    v = code_judge(lambda ns, out: ns.get("add")(2, 3) == 5)
    assert v("write add function", answer) == 1.0


def test_code_judge_fail_when_check_false():
    answer = "```python\ndef add(a, b): return a - b\n```"
    v = code_judge(lambda ns, out: ns.get("add")(2, 3) == 5)
    assert v("write add function", answer) == 0.0


def test_code_judge_syntax_error_returns_zero():
    answer = "```python\ndef broken(\n```"
    v = code_judge(lambda ns, out: True)
    assert v("q", answer) == 0.0


def test_code_judge_runtime_error_returns_zero():
    answer = "```python\nraise ValueError('oops')\n```"
    v = code_judge(lambda ns, out: True)
    assert v("q", answer) == 0.0


def test_code_judge_exception_in_check_returns_zero():
    answer = "```python\nx = 1\n```"
    def bad_check(ns, out):
        raise RuntimeError("check failed")
    v = code_judge(bad_check)
    assert v("q", answer) == 0.0


def test_code_judge_receives_stdout():
    answer = "```python\nprint('hello world')\n```"
    captured = []
    def check(ns, out):
        captured.append(out)
        return True
    v = code_judge(check)
    v("q", answer)
    assert "hello world" in captured[0]


def test_code_judge_receives_namespace():
    answer = "```python\nresult = 42\n```"
    ns_ref = []
    def check(ns, out):
        ns_ref.append(ns)
        return True
    v = code_judge(check)
    v("q", answer)
    assert ns_ref[0].get("result") == 42


def test_code_judge_function_definition_accessible():
    answer = "```python\ndef factorial(n):\n    return 1 if n <= 1 else n * factorial(n-1)\n```"
    v = code_judge(lambda ns, out: ns["factorial"](5) == 120)
    assert v("write factorial", answer) == 1.0


def test_code_judge_class_definition_accessible():
    answer = "```python\nclass Counter:\n    def __init__(self): self.n = 0\n    def inc(self): self.n += 1\n```"
    def check(ns, out):
        c = ns["Counter"]()
        c.inc(); c.inc()
        return c.n == 2
    assert code_judge(check)("q", answer) == 1.0


def test_code_judge_extracts_from_full_trace():
    trace = "I need to write a sort function.\n```python\ndef sort(lst): return sorted(lst)\n```\nANSWER: done"
    v = code_judge(lambda ns, out: ns["sort"]([3,1,2]) == [1,2,3])
    assert v("sort", trace) == 1.0


def test_code_judge_missing_function_fails():
    answer = "```python\nx = 99\n```"
    v = code_judge(lambda ns, out: ns["missing_function"]() == 0)
    assert v("q", answer) == 0.0


def test_code_judge_multiple_functions_in_block():
    answer = "```python\ndef double(x): return x * 2\ndef triple(x): return x * 3\n```"
    def check(ns, out):
        return ns["double"](5) == 10 and ns["triple"](4) == 12
    assert code_judge(check)("q", answer) == 1.0


def test_code_judge_with_print_and_namespace():
    answer = "```python\nval = 7\nprint('val is', val)\n```"
    results = {}
    def check(ns, out):
        results["ns"]  = ns.get("val")
        results["out"] = out
        return True
    code_judge(check)("q", answer)
    assert results["ns"] == 7
    assert "val is 7" in results["out"]


def test_code_judge_debugging_scenario():
    # simulate: agent debugs a broken binary search and produces a fixed version
    broken_then_fixed = (
        "I see the bug — `right` should be `len(arr)-1`, not `len(arr)`.\n"
        "```python\n"
        "def binary_search(arr, target):\n"
        "    left, right = 0, len(arr) - 1\n"
        "    while left <= right:\n"
        "        mid = (left + right) // 2\n"
        "        if arr[mid] == target: return mid\n"
        "        elif arr[mid] < target: left = mid + 1\n"
        "        else: right = mid - 1\n"
        "    return -1\n"
        "```\n"
        "ANSWER: fixed"
    )
    def check(ns, out):
        bs = ns.get("binary_search")
        return (bs and bs([1,2,3,4,5], 3) == 2
                    and bs([1,2,3,4,5], 6) == -1)
    assert code_judge(check)("debug binary search", broken_then_fixed) == 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# Forecaster — kNN baseline behaviour
# ═══════════════════════════════════════════════════════════════════════════════

def test_forecaster_predicts_and_gates():
    fc = Forecaster(_embed, k=1, pca_dim=0).fit(
        ["bad run", "bad attempt", "good run", "good attempt"], [0, 0, 1, 1])
    assert fc.predict_fail("a bad one") > 0.5
    assert fc.predict_fail("a good one") < 0.5
    assert fc.should_intervene("a bad one") and not fc.should_intervene("a good one")


def test_forecaster_add_grows_store():
    fc = Forecaster(_embed, k=1, pca_dim=0).fit(["good a"], [1])
    fc.add("bad b", 0)
    assert len(fc._labels) == 2 and fc.predict_fail("bad c") > 0.5


def test_forecaster_single_class_all_success_returns_low_fail():
    fc = Forecaster(_embed, k=3, pca_dim=0).fit(["good a", "good b", "good c"], [1, 1, 1])
    assert fc.predict_fail("good d") == 0.0


def test_forecaster_single_class_abstains():
    # only one outcome class seen → no discriminative signal → abstain (0.0),
    # never cascade into retrying everything. True for all-fail AND all-pass.
    fc_fail = Forecaster(_embed, k=3, pca_dim=0).fit(["bad a", "bad b", "bad c"], [0, 0, 0])
    assert fc_fail.predict_fail("bad d") == 0.0
    fc_pass = Forecaster(_embed, k=3, pca_dim=0).fit(["good a", "good b", "good c"], [1, 1, 1])
    assert fc_pass.predict_fail("good d") == 0.0


def test_forecaster_predict_fail_bounded():
    fc = Forecaster(_embed, k=2, pca_dim=0).fit(
        ["bad x", "bad y", "good x", "good y"], [0, 0, 1, 1])
    for trace in ["bad z", "good z", "neutral"]:
        p = fc.predict_fail(trace)
        assert 0.0 <= p <= 1.0


def test_forecaster_should_intervene_threshold():
    fc = Forecaster(_embed, k=1, pca_dim=0).fit(["bad run", "good run"], [0, 1])
    assert fc.should_intervene("bad trace", threshold=0.5)
    assert not fc.should_intervene("good trace", threshold=0.5)
    assert not fc.should_intervene("bad trace", threshold=1.1)


def test_forecaster_explain_returns_neighbors():
    fc = Forecaster(_embed, k=2, pca_dim=0).fit(
        ["bad run", "bad fail", "good run", "good pass"], [0, 0, 1, 1])
    neighbors = fc.explain("bad query", k=2)
    assert len(neighbors) == 2
    for n in neighbors:
        assert {"similarity", "label", "outcome", "excerpt"} <= set(n)


def test_forecaster_explain_bad_trace_finds_fail_neighbors():
    # use a 4-trace store with clear separation; pca_dim=0 keeps raw 2-d vecs
    fc = Forecaster(_embed, k=2, pca_dim=0).fit(
        ["bad run", "bad fail", "good run", "good pass"], [0, 0, 1, 1])
    neighbors = fc.explain("bad query", k=2)
    # "bad query" is nearest to the "bad" cluster -> both neighbors should be failures
    fail_labels = [n["label"] for n in neighbors]
    assert 0 in fail_labels   # at least one failure neighbor


def test_forecaster_explain_empty_store_returns_empty():
    fc = Forecaster(_embed, k=2, pca_dim=0)
    assert fc.explain("anything") == []


def test_forecaster_nearest_failure_returns_string():
    fc = Forecaster(_embed, k=2, pca_dim=0).fit(
        ["bad run", "good run"], [0, 1])
    result = fc.nearest_failure("bad trace")
    assert result is not None and isinstance(result, str)


def test_forecaster_nearest_failure_returns_none_when_no_failures():
    fc = Forecaster(_embed, k=2, pca_dim=0).fit(["good a", "good b"], [1, 1])
    assert fc.nearest_failure("anything") is None


def test_forecaster_fit_overwrites_previous_store():
    fc = Forecaster(_embed, k=1, pca_dim=0).fit(["bad a"], [0])
    fc.fit(["good x", "good y"], [1, 1])
    assert len(fc._labels) == 2 and all(l == 1 for l in fc._labels)


# ═══════════════════════════════════════════════════════════════════════════════
# Forecaster — PCA
# ═══════════════════════════════════════════════════════════════════════════════

def test_forecaster_pca_inactive_when_store_small():
    fc = Forecaster(_embed3d, k=1, pca_dim=4)
    fc.fit(["bad a", "good b", "neutral c"], [0, 1, 1])
    # 3 vecs <= pca_dim=4 → PCA should stay off
    assert fc._pca is None
    assert len(fc._vecs[0]) == 3   # raw 3-d


def test_forecaster_pca_activates_once_store_exceeds_pca_dim():
    fc = Forecaster(_embed3d, k=1, pca_dim=2)
    traces = ["bad a", "good b", "neutral c"]
    labels = [0, 1, 1]
    fc.fit(traces, labels)
    # 3 vecs > pca_dim=2 → PCA should be active
    assert fc._pca is not None
    assert len(fc._vecs[0]) == 2   # projected to 2-d


def test_forecaster_pca_add_triggers_activation():
    fc = Forecaster(_embed3d, k=1, pca_dim=2)
    fc.fit(["bad a", "good b"], [0, 1])
    assert fc._pca is None          # 2 <= pca_dim=2, not yet
    fc.add("neutral c", 1)
    assert fc._pca is not None      # 3 > 2, now active
    assert len(fc._vecs[0]) == 2


def test_forecaster_raw_vecs_always_stored():
    fc = Forecaster(_embed3d, k=1, pca_dim=2)
    fc.fit(["bad a", "good b", "neutral c"], [0, 1, 1])
    assert len(fc._raw_vecs) == 3
    assert len(fc._raw_vecs[0]) == 3   # always full 3-d


def test_forecaster_pca_predict_still_bounded():
    fc = Forecaster(_embed3d, k=1, pca_dim=2)
    fc.fit(["bad a", "good b", "bad c", "good d"], [0, 1, 0, 1])
    for t in ["bad x", "good x", "neutral"]:
        p = fc.predict_fail(t)
        assert 0.0 <= p <= 1.0, f"out of range for {t!r}: {p}"


def test_forecaster_pca_does_not_change_predictions_qualitatively():
    fc = Forecaster(_embed3d, k=1, pca_dim=2)
    fc.fit(["bad a", "bad b", "good c", "good d"], [0, 0, 1, 1])
    assert fc.predict_fail("bad query") > 0.5
    assert fc.predict_fail("good query") < 0.5


def test_forecaster_pca_explain_consistent_dimension():
    fc = Forecaster(_embed3d, k=2, pca_dim=2)
    fc.fit(["bad a", "good b", "neutral c", "bad d"], [0, 1, 1, 0])
    neighbors = fc.explain("bad query", k=2)
    # should not crash; similarity should be a valid float
    for n in neighbors:
        assert isinstance(n["similarity"], float)
        assert 0.0 <= n["similarity"] <= 1.01   # cosine can exceed 1.0 by tiny float error


def test_forecaster_pca_vecs_and_raw_vecs_same_length():
    fc = Forecaster(_embed3d, k=1, pca_dim=2)
    fc.fit(["bad a", "good b", "neutral c"], [0, 1, 1])
    assert len(fc._vecs) == len(fc._raw_vecs) == 3


def test_forecaster_pca_dim_zero_disables_pca():
    fc = Forecaster(_embed3d, k=1, pca_dim=0)
    fc.fit(["bad a", "good b", "bad c", "good d", "neutral"], [0, 1, 0, 1, 1])
    assert fc._pca is None
    assert len(fc._vecs[0]) == 3   # raw


def test_forecaster_pca_fit_then_add_keeps_store_consistent():
    fc = Forecaster(_embed3d, k=1, pca_dim=2)
    fc.fit(["bad a", "good b", "neutral c"], [0, 1, 1])
    fc.add("bad d", 0)
    assert len(fc._vecs) == len(fc._raw_vecs) == len(fc._labels) == 4


def test_forecaster_pca_custom_dim():
    def embed5d(texts):
        return np.eye(5)[:len(texts)]
    fc = Forecaster(embed5d, k=1, pca_dim=3)
    traces = [str(i) for i in range(6)]
    labels = [i % 2 for i in range(6)]
    fc.fit(traces, labels)
    assert fc._pca is not None
    assert len(fc._vecs[0]) == 3


# ═══════════════════════════════════════════════════════════════════════════════
# TaskResult / ComponentResult
# ═══════════════════════════════════════════════════════════════════════════════

def test_task_result_properties():
    cr_pass = ComponentResult("q1", "trace", label=1, p_fail=0.2, retried=False, neighbor=None)
    cr_fail = ComponentResult("q2", "trace", label=0, p_fail=0.8, retried=True,  neighbor="x")
    tr = TaskResult(task="t", components=[cr_pass, cr_fail])
    assert tr.n_pass == 1
    assert tr.n_fail == 1
    assert tr.n_intervened == 1


def test_task_result_empty_components():
    tr = TaskResult(task="empty")
    assert tr.n_pass == tr.n_fail == tr.n_intervened == 0


def test_task_result_summary_contains_task():
    cr = ComponentResult("q", "trace", 1, None, False, None)
    tr = TaskResult(task="my special task", components=[cr])
    assert "my special task" in tr.summary()


def test_task_result_summary_counts():
    crs = [ComponentResult(f"q{i}", "t", i % 2, None, False, None) for i in range(6)]
    tr = TaskResult(task="t", components=crs)
    s = tr.summary()
    assert "Pass: 3" in s and "Fail: 3" in s


def test_component_result_stores_all_fields():
    cr = ComponentResult("question", "trace text", label=0,
                         p_fail=0.9, retried=True, neighbor="near fail")
    assert cr.question == "question"
    assert cr.trace    == "trace text"
    assert cr.label    == 0
    assert cr.p_fail   == 0.9
    assert cr.retried  is True
    assert cr.neighbor == "near fail"


# ═══════════════════════════════════════════════════════════════════════════════
# run_task integration
# ═══════════════════════════════════════════════════════════════════════════════

def _simple_agent(prompt):
    if "Break the task" in prompt:
        return ("What is the answer?\nWhat is the reason?", 0)
    return ("ANSWER: 42", 0)


def test_run_task_default_threshold_is_none(monkeypatch):
    import inspect
    sig = inspect.signature(pipeline.run_task)
    assert sig.parameters["threshold"].default is None


def test_run_task_returns_task_result():
    result = pipeline.run_task("simple task", agent=_simple_agent, display=False)
    assert isinstance(result, TaskResult)
    assert len(result.components) >= 1


def test_run_task_no_verifier_labels_optimistic():
    result = pipeline.run_task("simple task", agent=_simple_agent, display=False)
    assert all(c.label == 1 for c in result.components)


def test_run_task_no_forecaster_skips_prediction():
    result = pipeline.run_task("simple task", agent=_simple_agent, display=False)
    assert all(c.p_fail is None for c in result.components)


def test_run_task_with_verifier_sets_label():
    result = pipeline.run_task(
        "simple task",
        agent=_simple_agent,
        verifier=lambda q, a: 1.0,
        display=False,
    )
    assert all(c.label == 1 for c in result.components)


def test_run_task_verifier_fail_sets_zero():
    result = pipeline.run_task(
        "simple task",
        agent=_simple_agent,
        verifier=lambda q, a: 0.0,
        display=False,
    )
    assert all(c.label == 0 for c in result.components)


def test_run_task_with_forecaster_stores_traces():
    fc = Forecaster(_embed, k=1, pca_dim=0)
    pipeline.run_task(
        "simple task",
        agent=_simple_agent,
        verifier=lambda q, a: 1.0,
        forecaster=fc,
        display=False,
    )
    assert len(fc._labels) >= 1


def test_run_task_cap_limits_components():
    result = pipeline.run_task(
        "simple task",
        agent=_simple_agent,
        cap=1,
        display=False,
    )
    assert len(result.components) == 1


def test_run_task_retry_false_never_retries():
    fc = Forecaster(_embed, k=1, pca_dim=0)
    fc.fit(
        ["bad a", "bad b", "good c", "good d",
         "bad e", "bad f", "good g", "good h"],
        [0, 0, 1, 1, 0, 0, 1, 1],
    )
    result = pipeline.run_task(
        "simple task",
        agent=lambda p: ("bad trace ANSWER: oops", 0) if "Break" not in p else ("What?", 0),
        verifier=lambda q, a: 0.0,
        forecaster=fc,
        retry=False,
        threshold=0.1,
        display=False,
    )
    assert all(not c.retried for c in result.components)


# ═══════════════════════════════════════════════════════════════════════════════
# tool_agent bail_fn interface (offline — no API call)
# ═══════════════════════════════════════════════════════════════════════════════

def test_tool_agent_has_bail_fn_attribute(monkeypatch):
    # tool_agent should expose bail_fn=None so run_task can wire mid-call exit
    # without an API key — just importing and calling the factory is enough.
    import sys
    # stub out the tools import so the factory doesn't need tools.py on path
    fake_tools = type(sys)("tools")
    fake_tools.TOOL_DEFINITIONS = []
    fake_tools.dispatch = lambda name, inp: ""
    monkeypatch.setitem(sys.modules, "tools", fake_tools)
    from agents import tool_agent
    a = tool_agent()
    assert hasattr(a, 'bail_fn') and a.bail_fn is None


def test_tool_agent_bail_fn_settable(monkeypatch):
    import sys
    fake_tools = type(sys)("tools")
    fake_tools.TOOL_DEFINITIONS = []
    fake_tools.dispatch = lambda name, inp: ""
    monkeypatch.setitem(sys.modules, "tools", fake_tools)
    from agents import tool_agent
    a = tool_agent()
    sentinel = lambda trace: False
    a.bail_fn = sentinel
    assert a.bail_fn is sentinel


# ═══════════════════════════════════════════════════════════════════════════════
# make_retriever
# ═══════════════════════════════════════════════════════════════════════════════

def _unit_embedder(texts):
    vecs = []
    for t in texts:
        h = hash(t[0]) % 100 / 100.0 if t else 0.0
        vecs.append([h, 1.0 - h])
    v = np.array(vecs, dtype="float32")
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    return v / (norms + 1e-9)


def test_make_retriever_returns_string():
    retrieve = make_retriever(["alpha chunk", "beta chunk"], _unit_embedder)
    assert isinstance(retrieve("alpha chunk"), str)


def test_make_retriever_empty_corpus():
    retrieve = make_retriever([], _unit_embedder)
    assert retrieve("anything") == ""


def test_make_retriever_whitespace_only_chunks_ignored():
    retrieve = make_retriever(["   ", "\n", "real content here"], _unit_embedder)
    assert "real content here" in retrieve("real content here")


def test_make_retriever_respects_word_budget():
    chunks = ["word " * 10, "other " * 10, "third " * 10]
    retrieve = make_retriever(chunks, _unit_embedder)
    result = retrieve("word", words=12)
    assert len(result.split()) <= 15


def test_make_retriever_joins_multiple_chunks():
    chunks = ["aaa aaa", "bbb bbb", "ccc ccc"]
    retrieve = make_retriever(chunks, _unit_embedder)
    result = retrieve("aaa", words=100)
    assert "\n\n" in result


def test_make_retriever_single_chunk():
    retrieve = make_retriever(["only this chunk"], _unit_embedder)
    assert "only this chunk" in retrieve("only", words=100)


def test_make_retriever_non_overlapping_budgets():
    chunks = ["apple banana cherry", "dog elephant fox"]
    retrieve = make_retriever(chunks, _unit_embedder)
    result = retrieve("apple", words=3)
    # only the most relevant chunk should fit within 3 words
    assert len(result.split()) <= 5
