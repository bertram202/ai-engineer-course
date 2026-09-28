import math

import numpy as np
import pytest
from ex04_information import (
    cross_entropy,
    entropy,
    kl_divergence,
    log_softmax,
    perplexity,
    softmax,
    top_p_filter,
)


def test_softmax_sums_to_one_and_keeps_order():
    p = softmax(np.array([1.0, 3.0, 2.0]))
    assert p.sum() == pytest.approx(1.0)
    assert np.argmax(p) == 1 and p[0] < p[2] < p[1]


def test_softmax_is_stable():
    p = softmax(np.array([1000.0, 1001.0, 1002.0]))
    assert np.all(np.isfinite(p))
    assert np.allclose(p, softmax(np.array([0.0, 1.0, 2.0])))


def test_softmax_temperature():
    z = np.array([2.0, 1.0, 0.0])
    assert softmax(z, 0.05)[0] > 0.999          # T → 0: почти argmax
    assert np.allclose(softmax(z, 1e4), 1 / 3, atol=1e-3)  # T → ∞: почти равномерно
    assert np.allclose(softmax(z, 2.0), softmax(z / 2))
    with pytest.raises(ValueError):
        softmax(z, 0)


def test_log_softmax():
    z = np.array([0.5, -1.0, 2.0])
    assert np.allclose(log_softmax(z), np.log(softmax(z)))
    big = log_softmax(np.array([1000.0, 0.0]))
    assert np.all(np.isfinite(big)) and big[0] == pytest.approx(0.0) and big[1] == pytest.approx(-1000.0)


def test_entropy():
    assert entropy(np.array([0.25] * 4)) == pytest.approx(math.log(4))
    assert entropy(np.array([1.0, 0.0, 0.0])) == pytest.approx(0.0)
    assert entropy(np.array([0.5, 0.5, 0.0])) == pytest.approx(math.log(2))


def test_cross_entropy_and_kl():
    p = np.array([0.5, 0.3, 0.2])
    q = np.array([0.2, 0.5, 0.3])
    assert cross_entropy(p, q) >= entropy(p)
    assert kl_divergence(p, q) == pytest.approx(cross_entropy(p, q) - entropy(p))
    assert kl_divergence(p, p) == pytest.approx(0.0, abs=1e-12)
    assert kl_divergence(p, q) != pytest.approx(kl_divergence(q, p))
    assert kl_divergence(p, q) == pytest.approx(sum(a * math.log(a / b) for a, b in zip(p, q)))


def test_cross_entropy_with_one_hot_target():
    target = np.array([0.0, 1.0, 0.0])
    q = np.array([0.1, 0.25, 0.65])
    assert cross_entropy(target, q) == pytest.approx(-math.log(0.25))


def test_perplexity():
    assert perplexity([math.log(0.25)] * 10) == pytest.approx(4.0)
    assert perplexity([0.0, 0.0]) == pytest.approx(1.0)
    assert perplexity(np.log([0.5, 0.125])) == pytest.approx(4.0)  # среднее геометрическое


def test_top_p_filter():
    probs = np.array([0.1, 0.5, 0.05, 0.3, 0.05])
    kept = top_p_filter(probs, 0.75)
    assert np.allclose(kept, [0.0, 0.5 / 0.8, 0.0, 0.3 / 0.8, 0.0])
    assert np.allclose(top_p_filter(probs, 0.8), kept)  # 0.5 + 0.3 = 0.8 — уже достаточно
    assert np.count_nonzero(top_p_filter(probs, 0.85)) == 3
    assert np.allclose(top_p_filter(probs, 1.0), probs)
    assert top_p_filter(probs, 0.01).tolist() == [0.0, 1.0, 0.0, 0.0, 0.0]
