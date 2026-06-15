"""Offline gates for the forecasting core — no API key needed."""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import forecast
from forecast import auc, knn_predict, cosine


def test_auc_perfect_chance_and_reversed():
    assert auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0      # perfect ranking
    assert auc([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1]) == 0.0      # exactly wrong
    assert auc([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5]) == 0.5      # constant -> chance


def test_auc_nan_when_one_class_missing():
    import math
    assert math.isnan(auc([1, 1, 1], [0.2, 0.5, 0.9]))


def test_knn_predicts_separable_outcomes():
    # two clusters; label tracks the first coordinate -> neighbours share outcome
    vecs = [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]]
    labels = [1, 1, 0, 0]
    preds = knn_predict(vecs, labels, k=1)
    assert auc(labels, preds) == 1.0          # LOO neighbour predicts outcome perfectly


def test_knn_predictions_are_bounded_probabilities():
    # predictions are means of {0,1} labels -> always in [0,1], one per input
    vecs = [[1.0, 0.0], [0.2, 0.9], [0.8, 0.3], [0.1, 1.0], [0.9, 0.1]]
    labels = [1, 0, 1, 0, 1]
    preds = knn_predict(vecs, labels, k=2)
    assert len(preds) == len(vecs)
    assert all(0.0 <= p <= 1.0 for p in preds)


def test_cosine_basic():
    assert abs(cosine([1, 0], [1, 0]) - 1.0) < 1e-9
    assert abs(cosine([1, 0], [0, 1]) - 0.0) < 1e-9
