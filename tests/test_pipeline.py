"""Offline gates for the public API — no API key needed."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pipeline
from pipeline import Forecaster, decompose, attempt, gold_judge


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
