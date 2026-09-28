"""Упражнение 3. Свёртка с нуля.

Теория: раздел «Свёртки». Урок 2.
Как в PyTorch, «свёртка» — это на самом деле взаимная корреляция: ядро не переворачивается.
"""

import numpy as np


def conv_output_size(n: int, kernel: int, stride: int = 1, padding: int = 0) -> int:
    """Размер выхода по одной оси: ⌊(n + 2p − k) / s⌋ + 1."""
    raise NotImplementedError("TODO")


def conv_params(c_in: int, c_out: int, kernel: int, bias: bool = True) -> int:
    """Число параметров свёрточного слоя c_in → c_out с квадратным ядром kernel × kernel."""
    raise NotImplementedError("TODO")


def conv2d(x: np.ndarray, weight: np.ndarray, bias: np.ndarray, stride: int = 1, padding: int = 0) -> np.ndarray:
    """Свёртка одного изображения.

    x: [C_in, H, W]; weight: [C_out, C_in, k, k]; bias: [C_out] → выход [C_out, H_out, W_out].
    Вход дополняется нулями по краям на `padding` пикселей.
    """
    raise NotImplementedError("TODO: np.pad, затем в каждой позиции (i, j) — сумма окна, умноженного на ядро")


def receptive_field(kernels: list[int], strides: list[int]) -> int:
    """Рецептивное поле нейрона последнего слоя в стопке свёрток (размер окна исходной картинки).

    Каждый слой с ядром k расширяет поле на (k − 1) · J, где J — произведение шагов предыдущих слоёв.
    """
    raise NotImplementedError("TODO")
