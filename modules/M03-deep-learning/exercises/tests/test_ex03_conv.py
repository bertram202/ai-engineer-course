import numpy as np
import pytest
import torch
from ex03_conv import conv2d, conv_output_size, conv_params, receptive_field


@pytest.mark.parametrize(("n", "k", "s", "p", "expected"), [
    (32, 5, 1, 0, 28), (32, 3, 1, 1, 32), (224, 7, 2, 3, 112), (28, 2, 2, 0, 14), (7, 3, 2, 0, 3),
])
def test_output_size(n, k, s, p, expected):
    assert conv_output_size(n, k, s, p) == expected


def test_conv_params():
    assert conv_params(3, 16, 5) == 1216
    assert conv_params(64, 128, 3, bias=False) == 73_728


@pytest.mark.parametrize(("stride", "padding"), [(1, 0), (1, 1), (2, 1), (3, 2)])
def test_conv2d_matches_pytorch(stride, padding):
    rng = np.random.default_rng(0)
    x = rng.normal(size=(3, 9, 11))
    w = rng.normal(size=(4, 3, 3, 3))
    b = rng.normal(size=4)
    ours = conv2d(x, w, b, stride=stride, padding=padding)
    theirs = torch.nn.functional.conv2d(torch.tensor(x)[None], torch.tensor(w), torch.tensor(b),
                                        stride=stride, padding=padding)[0].numpy()
    assert ours.shape == theirs.shape
    assert np.allclose(ours, theirs)


def test_conv2d_edge_detector():
    x = np.zeros((1, 4, 4))
    x[0, :, 2:] = 1.0  # вертикальная граница между столбцами 1 и 2
    w = np.array([[[[-1.0, 1.0], [-1.0, 1.0]]]])  # разность соседей по горизонтали
    out = conv2d(x, w, np.zeros(1))
    assert out.shape == (1, 3, 3)
    assert np.array_equal(out[0, :, 1], [2.0, 2.0, 2.0])  # граница найдена в каждой строке
    assert out[0, :, 0].sum() == 0 and out[0, :, 2].sum() == 0


def test_receptive_field():
    assert receptive_field([3], [1]) == 3
    assert receptive_field([3, 3], [1, 1]) == 5
    assert receptive_field([3, 3, 3], [1, 1, 1]) == 7
    assert receptive_field([3, 2, 3], [1, 2, 1]) == 8  # после пулинга с шагом 2 поле растёт быстрее
