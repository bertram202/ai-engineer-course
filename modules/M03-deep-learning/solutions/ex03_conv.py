"""Упражнение 3. Свёртка с нуля.

Теория: раздел «Свёртки». Урок 2.
Как в PyTorch, «свёртка» — это на самом деле взаимная корреляция: ядро не переворачивается.
"""

import numpy as np


def conv_output_size(n: int, kernel: int, stride: int = 1, padding: int = 0) -> int:
    """Размер выхода по одной оси: ⌊(n + 2p − k) / s⌋ + 1."""
    return (n + 2 * padding - kernel) // stride + 1


def conv_params(c_in: int, c_out: int, kernel: int, bias: bool = True) -> int:
    """Число параметров свёрточного слоя c_in → c_out с квадратным ядром kernel × kernel."""
    return c_out * (c_in * kernel * kernel + (1 if bias else 0))


def conv2d(x: np.ndarray, weight: np.ndarray, bias: np.ndarray, stride: int = 1, padding: int = 0) -> np.ndarray:
    """Свёртка одного изображения.

    x: [C_in, H, W]; weight: [C_out, C_in, k, k]; bias: [C_out] → выход [C_out, H_out, W_out].
    Вход дополняется нулями по краям на `padding` пикселей.
    """
    # подсказка: np.pad, затем в каждой позиции (i, j) — сумма окна, умноженного на ядро
    c_out, _, k, _ = weight.shape
    x = np.pad(x, ((0, 0), (padding, padding), (padding, padding)))
    h_out = conv_output_size(x.shape[1], k, stride)
    w_out = conv_output_size(x.shape[2], k, stride)
    out = np.empty((c_out, h_out, w_out))
    for i in range(h_out):
        for j in range(w_out):
            window = x[:, i * stride : i * stride + k, j * stride : j * stride + k]  # [C_in, k, k]
            out[:, i, j] = np.tensordot(weight, window, axes=([1, 2, 3], [0, 1, 2])) + bias
    return out


def receptive_field(kernels: list[int], strides: list[int]) -> int:
    """Рецептивное поле нейрона последнего слоя в стопке свёрток (размер окна исходной картинки).

    Каждый слой с ядром k расширяет поле на (k − 1) · J, где J — произведение шагов предыдущих слоёв.
    """
    field, jump = 1, 1
    for k, s in zip(kernels, strides):
        field += (k - 1) * jump
        jump *= s
    return field
