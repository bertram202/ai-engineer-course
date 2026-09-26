import numpy as np
import pytest
from ex01_metrics import (
    confusion_counts,
    expected_calibration_error,
    precision_recall_f1,
    roc_auc,
    threshold_for_recall,
)
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

Y = np.array([1, 1, 1, 0, 0, 0, 0, 1, 0, 0])
P = np.array([1, 0, 1, 1, 0, 0, 0, 1, 0, 0])


def test_confusion_counts():
    assert confusion_counts(Y, P) == {"tp": 3, "fp": 1, "fn": 1, "tn": 5}
    assert all(isinstance(v, int) for v in confusion_counts(Y, P).values())


def test_precision_recall_f1_match_sklearn():
    rng = np.random.default_rng(0)
    for _ in range(5):
        y, p = rng.integers(0, 2, 50), rng.integers(0, 2, 50)
        assert precision_recall_f1(y, p) == pytest.approx(
            (precision_score(y, p), recall_score(y, p), f1_score(y, p)))


def test_no_positive_predictions_gives_zero():
    assert precision_recall_f1(np.array([1, 0, 1]), np.array([0, 0, 0])) == (0.0, 0.0, 0.0)


def test_roc_auc_matches_sklearn_with_ties():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 300)
    scores = np.round(rng.random(300) + 0.5 * y, 1)  # округление создаёт много равных оценок
    assert roc_auc(y, scores) == pytest.approx(roc_auc_score(y, scores))


def test_roc_auc_extremes():
    y = np.array([0, 0, 1, 1])
    assert roc_auc(y, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert roc_auc(y, np.array([0.9, 0.8, 0.2, 0.1])) == 0.0
    assert roc_auc(y, np.array([0.5, 0.5, 0.5, 0.5])) == 0.5


def test_threshold_for_recall():
    y = np.array([1, 0, 1, 0, 1, 0, 1, 0])
    s = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2])
    assert threshold_for_recall(y, s, 0.5) == 0.7    # находим 2 из 4
    assert threshold_for_recall(y, s, 0.75) == 0.5   # 3 из 4
    assert threshold_for_recall(y, s, 1.0) == 0.3


def test_expected_calibration_error():
    y = np.array([1, 0, 1, 1, 0, 0])
    assert expected_calibration_error(y, np.array([0.8, 0.8, 0.8, 0.2, 0.2, 0.2]), n_bins=10) == pytest.approx(
        0.5 * abs(0.8 - 2 / 3) + 0.5 * abs(0.2 - 1 / 3))
    perfect = np.array([1.0, 0.0, 1.0, 1.0, 0.0, 0.0])
    assert expected_calibration_error(y, perfect) == pytest.approx(0.0)  # 1.0 попадает в последнюю корзину
