"""Offline gates for the forecasting core — no API key needed."""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import forecast
from forecast import auc, knn_predict, knn_predict_cross, spearman, cosine


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


def test_spearman_monotonic_and_inverse():
    assert abs(spearman([1, 2, 3, 4], [10, 20, 30, 40]) - 1.0) < 1e-9
    assert abs(spearman([1, 2, 3, 4], [40, 30, 20, 10]) + 1.0) < 1e-9


def test_knn_cross_predicts_from_train_set():
    train_v = [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]]
    train_y = [1, 1, 0, 0]
    test_v = [[0.95, 0.05], [0.05, 0.95]]      # near class-1 cluster, near class-0 cluster
    preds = knn_predict_cross(train_v, train_y, test_v, k=1)
    assert preds[0] > preds[1]                 # first looks like success, second like failure


def test_cosine_basic():
    assert abs(cosine([1, 0], [1, 0]) - 1.0) < 1e-9
    assert abs(cosine([1, 0], [0, 1]) - 0.0) < 1e-9


def test_cosine_zero_vector_no_crash():
    # denominator guard (1e-12) should prevent division by zero
    result = cosine([0, 0], [1, 0])
    assert result == 0.0


def test_knn_predict_single_item():
    # LOO with n=1: no neighbours -> fallback to base rate
    preds = knn_predict([[1.0, 0.0]], [1], k=1)
    assert len(preds) == 1
    assert 0.0 <= preds[0] <= 1.0


def test_knn_predict_k_larger_than_n_minus_1():
    # k > available neighbours: should not crash, just use all available
    vecs   = [[1.0, 0.0], [0.0, 1.0]]
    labels = [1, 0]
    preds = knn_predict(vecs, labels, k=100)
    assert len(preds) == 2
    assert all(0.0 <= p <= 1.0 for p in preds)


def test_knn_cross_k_larger_than_train():
    train_v = [[1.0, 0.0], [0.0, 1.0]]
    train_y = [1, 0]
    test_v  = [[0.9, 0.1]]
    preds = knn_predict_cross(train_v, train_y, test_v, k=1000)
    assert len(preds) == 1
    assert 0.0 <= preds[0] <= 1.0


def test_knn_cross_empty_test_set():
    train_v = [[1.0, 0.0], [0.0, 1.0]]
    train_y = [1, 0]
    preds = knn_predict_cross(train_v, train_y, [], k=1)
    assert preds == []


def test_auc_single_positive_negative_pair():
    assert auc([0, 1], [0.2, 0.8]) == 1.0
    assert auc([0, 1], [0.8, 0.2]) == 0.0


def test_auc_tie_scores_half_credit():
    # tied score between pos and neg -> 0.5 credit each -> AUC = 0.5
    assert auc([0, 1], [0.5, 0.5]) == 0.5


def test_spearman_constant_input_no_crash():
    # all same values -> zero variance -> result near 0 (not NaN/exception)
    result = spearman([1, 1, 1], [2, 3, 4])
    assert isinstance(result, float)


def test_spearman_length_one_no_crash():
    result = spearman([1.0], [1.0])
    assert isinstance(result, float)


def test_knn_predict_all_same_label():
    vecs   = [[1.0, 0.0], [0.9, 0.1], [0.8, 0.2]]
    labels = [1, 1, 1]
    preds  = knn_predict(vecs, labels, k=2)
    assert all(p == 1.0 for p in preds)


def test_knn_cross_perfect_transfer():
    # train on two clean clusters; test points land exactly in each cluster
    train_v = [[1.0, 0.0], [0.95, 0.05], [0.0, 1.0], [0.05, 0.95]]
    train_y = [1, 1, 0, 0]
    test_v  = [[0.98, 0.02], [0.02, 0.98]]
    preds   = knn_predict_cross(train_v, train_y, test_v, k=2)
    assert preds[0] > 0.5 and preds[1] < 0.5
