"""Offline gates for the forecasting core — no API key needed."""
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import trace_use.forecast as forecast
from trace_use.forecast import auc, knn_predict, knn_predict_cross, spearman, cosine


# ═══════════════════════════════════════════════════════════════════════════════
# auc
# ═══════════════════════════════════════════════════════════════════════════════

def test_auc_perfect_ranking():
    assert auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0


def test_auc_worst_ranking():
    assert auc([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1]) == 0.0


def test_auc_chance_level():
    assert auc([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5]) == 0.5


def test_auc_nan_when_single_class():
    assert math.isnan(auc([1, 1, 1], [0.2, 0.5, 0.9]))
    assert math.isnan(auc([0, 0, 0], [0.1, 0.2, 0.3]))


def test_auc_single_positive_negative_pair_perfect():
    assert auc([0, 1], [0.2, 0.8]) == 1.0


def test_auc_single_positive_negative_pair_worst():
    assert auc([0, 1], [0.8, 0.2]) == 0.0


def test_auc_tied_scores_half_credit():
    assert auc([0, 1], [0.5, 0.5]) == 0.5


def test_auc_many_pairs_sorted():
    n = 20
    labels = [0] * n + [1] * n
    scores = list(range(n)) + list(range(n, 2 * n))  # positives score higher
    assert auc(labels, scores) == 1.0


def test_auc_all_positives_higher_than_all_negatives():
    labels = [0, 0, 0, 1, 1, 1]
    scores = [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]
    assert auc(labels, scores) == 1.0


def test_auc_symmetric_about_half():
    labels = [0, 1]
    assert auc(labels, [0.3, 0.7]) == 1.0
    assert auc(labels, [0.7, 0.3]) == 0.0


def test_auc_returns_float():
    result = auc([0, 1], [0.4, 0.6])
    assert isinstance(result, float)


# ═══════════════════════════════════════════════════════════════════════════════
# cosine
# ═══════════════════════════════════════════════════════════════════════════════

def test_cosine_identical_vectors():
    assert abs(cosine([1, 0], [1, 0]) - 1.0) < 1e-9


def test_cosine_orthogonal_vectors():
    assert abs(cosine([1, 0], [0, 1]) - 0.0) < 1e-9


def test_cosine_antiparallel_vectors():
    result = cosine([1, 0], [-1, 0])
    assert result < 0   # antiparallel -> negative cosine


def test_cosine_zero_vector_no_crash():
    result = cosine([0, 0], [1, 0])
    assert result == 0.0


def test_cosine_both_zero_no_crash():
    result = cosine([0, 0], [0, 0])
    assert isinstance(result, float)


def test_cosine_normalized_vectors():
    # unit vectors at 60° -> cosine = 0.5
    v1 = [1.0, 0.0]
    v2 = [0.5, math.sqrt(3) / 2]
    assert abs(cosine(v1, v2) - 0.5) < 1e-6


def test_cosine_high_dimensional():
    np.random.seed(0)
    v1 = np.random.randn(512).tolist()
    v2 = v1[:]   # identical
    assert abs(cosine(v1, v2) - 1.0) < 1e-5


def test_cosine_negative_components():
    v1 = [-1.0, 0.0]
    v2 = [-1.0, 0.0]
    assert abs(cosine(v1, v2) - 1.0) < 1e-9


def test_cosine_scaled_vectors_same_direction():
    # scaling should not change cosine similarity
    assert abs(cosine([2.0, 0.0], [5.0, 0.0]) - 1.0) < 1e-9


# ═══════════════════════════════════════════════════════════════════════════════
# knn_predict (LOO)
# ═══════════════════════════════════════════════════════════════════════════════

def test_knn_predict_separable_clusters():
    vecs   = [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]]
    labels = [1, 1, 0, 0]
    preds  = knn_predict(vecs, labels, k=1)
    assert auc(labels, preds) == 1.0


def test_knn_predict_returns_probabilities():
    vecs   = [[1.0, 0.0], [0.2, 0.9], [0.8, 0.3], [0.1, 1.0], [0.9, 0.1]]
    labels = [1, 0, 1, 0, 1]
    preds  = knn_predict(vecs, labels, k=2)
    assert len(preds) == 5
    assert all(0.0 <= p <= 1.0 for p in preds)


def test_knn_predict_single_item():
    preds = knn_predict([[1.0, 0.0]], [1], k=1)
    assert len(preds) == 1
    assert 0.0 <= preds[0] <= 1.0


def test_knn_predict_k_larger_than_n_minus_1():
    vecs   = [[1.0, 0.0], [0.0, 1.0]]
    labels = [1, 0]
    preds  = knn_predict(vecs, labels, k=100)
    assert len(preds) == 2
    assert all(0.0 <= p <= 1.0 for p in preds)


def test_knn_predict_all_same_label_returns_one():
    vecs   = [[1.0, 0.0], [0.9, 0.1], [0.8, 0.2]]
    labels = [1, 1, 1]
    preds  = knn_predict(vecs, labels, k=2)
    assert all(p == 1.0 for p in preds)


def test_knn_predict_all_fail_returns_zero():
    vecs   = [[1.0, 0.0], [0.9, 0.1], [0.8, 0.2]]
    labels = [0, 0, 0]
    preds  = knn_predict(vecs, labels, k=2)
    assert all(p == 0.0 for p in preds)


def test_knn_predict_k1_exact_neighbour():
    # k=1 LOO: each item's prediction = label of its nearest neighbour
    vecs   = [[1.0, 0.0], [0.95, 0.05], [0.0, 1.0], [0.05, 0.95]]
    labels = [1, 1, 0, 0]
    preds  = knn_predict(vecs, labels, k=1)
    assert preds[0] == 1.0   # nearest to [1,0] is [0.95,0.05] -> label 1
    assert preds[2] == 0.0   # nearest to [0,1] is [0.05,0.95] -> label 0


def test_knn_predict_k2_averages_two_neighbours():
    # two neighbours with labels [1, 0] -> prediction = 0.5
    vecs   = [[1.0, 0.0], [0.9, 0.1], [0.8, 0.2], [0.0, 1.0]]
    labels = [1, 0, 1, 0]
    preds  = knn_predict(vecs, labels, k=2)
    assert all(0.0 <= p <= 1.0 for p in preds)


def test_knn_predict_returns_list():
    preds = knn_predict([[1.0, 0.0], [0.0, 1.0]], [1, 0], k=1)
    assert isinstance(preds, list)


def test_knn_predict_large_dataset():
    np.random.seed(42)
    n = 100
    vecs   = np.random.randn(n, 10).tolist()
    labels = [i % 2 for i in range(n)]
    preds  = knn_predict(vecs, labels, k=5)
    assert len(preds) == n
    assert all(0.0 <= p <= 1.0 for p in preds)


# ═══════════════════════════════════════════════════════════════════════════════
# knn_predict_cross
# ═══════════════════════════════════════════════════════════════════════════════

def test_knn_cross_separable_clusters():
    train_v = [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]]
    train_y = [1, 1, 0, 0]
    test_v  = [[0.95, 0.05], [0.05, 0.95]]
    preds   = knn_predict_cross(train_v, train_y, test_v, k=1)
    assert preds[0] > preds[1]


def test_knn_cross_k_larger_than_train():
    train_v = [[1.0, 0.0], [0.0, 1.0]]
    train_y = [1, 0]
    test_v  = [[0.9, 0.1]]
    preds   = knn_predict_cross(train_v, train_y, test_v, k=1000)
    assert len(preds) == 1 and 0.0 <= preds[0] <= 1.0


def test_knn_cross_empty_test_set():
    train_v = [[1.0, 0.0], [0.0, 1.0]]
    train_y = [1, 0]
    assert knn_predict_cross(train_v, train_y, [], k=1) == []


def test_knn_cross_perfect_transfer():
    train_v = [[1.0, 0.0], [0.95, 0.05], [0.0, 1.0], [0.05, 0.95]]
    train_y = [1, 1, 0, 0]
    test_v  = [[0.98, 0.02], [0.02, 0.98]]
    preds   = knn_predict_cross(train_v, train_y, test_v, k=2)
    assert preds[0] > 0.5 and preds[1] < 0.5


def test_knn_cross_k1_nearest_neighbor_exact():
    train_v = [[1.0, 0.0], [0.0, 1.0]]
    train_y = [1, 0]
    test_v  = [[0.99, 0.01]]   # nearest to train[0]
    preds   = knn_predict_cross(train_v, train_y, test_v, k=1)
    assert preds[0] == 1.0


def test_knn_cross_k2_mean_of_two():
    train_v = [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]]
    train_y = [1, 0, 1, 0]   # alternating
    test_v  = [[1.0, 0.0]]   # nearest are [1,0]=1 and [0.9,0.1]=0
    preds   = knn_predict_cross(train_v, train_y, test_v, k=2)
    assert abs(preds[0] - 0.5) < 1e-9


def test_knn_cross_multiple_test_points():
    train_v = [[1.0, 0.0], [0.0, 1.0]]
    train_y = [1, 0]
    test_v  = [[0.9, 0.1], [0.1, 0.9], [0.8, 0.2]]
    preds   = knn_predict_cross(train_v, train_y, test_v, k=1)
    assert len(preds) == 3
    assert preds[0] == 1.0 and preds[1] == 0.0 and preds[2] == 1.0


def test_knn_cross_high_dimensional():
    np.random.seed(7)
    train_v = np.random.randn(20, 64).tolist()
    train_y = [i % 2 for i in range(20)]
    test_v  = np.random.randn(5, 64).tolist()
    preds   = knn_predict_cross(train_v, train_y, test_v, k=3)
    assert len(preds) == 5
    assert all(0.0 <= p <= 1.0 for p in preds)


def test_knn_cross_all_same_label_in_train():
    train_v = [[1.0, 0.0], [0.9, 0.1], [0.8, 0.2]]
    train_y = [1, 1, 1]
    test_v  = [[0.5, 0.5]]
    preds   = knn_predict_cross(train_v, train_y, test_v, k=2)
    assert preds[0] == 1.0


def test_knn_cross_returns_list():
    preds = knn_predict_cross([[1.0, 0.0]], [1], [[0.5, 0.5]], k=1)
    assert isinstance(preds, list)


# ═══════════════════════════════════════════════════════════════════════════════
# spearman
# ═══════════════════════════════════════════════════════════════════════════════

def test_spearman_perfect_positive():
    assert abs(spearman([1, 2, 3, 4], [10, 20, 30, 40]) - 1.0) < 1e-9


def test_spearman_perfect_negative():
    assert abs(spearman([1, 2, 3, 4], [40, 30, 20, 10]) + 1.0) < 1e-9


def test_spearman_constant_input_no_crash():
    result = spearman([1, 1, 1], [2, 3, 4])
    assert isinstance(result, float)


def test_spearman_length_one_no_crash():
    result = spearman([1.0], [1.0])
    assert isinstance(result, float)


def test_spearman_returns_float():
    result = spearman([1, 2, 3], [3, 1, 2])
    assert isinstance(result, float)


def test_spearman_bounded_minus_one_to_one():
    result = spearman([5, 3, 1, 4, 2], [2, 4, 1, 3, 5])
    assert -1.0 <= result <= 1.0


def test_spearman_identical_sequences():
    result = spearman([1, 2, 3, 4, 5], [1, 2, 3, 4, 5])
    assert abs(result - 1.0) < 1e-9


def test_spearman_integer_and_float_inputs():
    r1 = spearman([1, 2, 3], [1.0, 2.0, 3.0])
    assert abs(r1 - 1.0) < 1e-9


def test_spearman_two_elements():
    assert abs(spearman([1, 2], [1, 2]) - 1.0) < 1e-9
    assert abs(spearman([1, 2], [2, 1]) + 1.0) < 1e-9
