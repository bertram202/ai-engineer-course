import numpy as np
import pytest
from ex02_svd import approx_error, lora_params, lora_share, low_rank_approx, rank_for_energy


def test_low_rank_has_right_rank_and_shape():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(20, 12))
    a3 = low_rank_approx(a, 3)
    assert a3.shape == a.shape
    assert np.linalg.matrix_rank(a3) == 3


def test_low_rank_recovers_low_rank_matrix():
    rng = np.random.default_rng(1)
    a = rng.normal(size=(30, 2)) @ rng.normal(size=(2, 25))
    assert np.allclose(low_rank_approx(a, 2), a)


def test_eckart_young_error():
    rng = np.random.default_rng(2)
    a = rng.normal(size=(15, 10))
    s = np.linalg.svd(a, compute_uv=False)
    for k in [1, 4, 9, 10]:
        direct = np.linalg.norm(a - low_rank_approx(a, k))
        assert approx_error(s, k) == pytest.approx(direct, abs=1e-9)


def test_rank_for_energy():
    s = np.array([3.0, 2.0, 1.0, 1.0, 1.0])  # квадраты: 9, 4, 1, 1, 1 → сумма 16
    assert rank_for_energy(s, 0.5) == 1   # 9/16 = 0.5625
    assert rank_for_energy(s, 0.8) == 2   # 13/16 = 0.8125
    assert rank_for_energy(s, 0.8125) == 2  # ровно на границе
    assert rank_for_energy(s, 1.0) == 5
    assert isinstance(rank_for_energy(s, 0.5), int)


def test_lora_params():
    assert lora_params(4096, 4096, 16) == 131_072
    assert lora_params(4096, 1024, 8) == 40_960


def test_lora_share_qwen3_attention():
    shapes = [(4096, 4096), (4096, 1024), (4096, 1024), (4096, 4096)]
    assert lora_share(shapes, 16) == pytest.approx(0.01015625)  # 425 984 / 41 943 040
