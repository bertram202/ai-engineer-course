"""Помощники для PyTorch: выбор устройства и воспроизводимость.

torch импортируется внутри функций: библиотека курса не тянет PyTorch в модули, где он не нужен.
"""

from __future__ import annotations

import os
import random
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import torch


def get_device() -> torch.device:
    """Лучшее доступное устройство: MPS (Apple Silicon) → CUDA → CPU.

    Переменная `MLCOURSE_DEVICE` (например, `cpu`) переопределяет выбор.
    """
    import torch

    if name := os.environ.get("MLCOURSE_DEVICE"):
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def seed_everything(seed: int = 42) -> None:
    """Зафиксировать генераторы случайных чисел Python, NumPy и PyTorch (CPU, CUDA, MPS)."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)  # заодно сидирует CUDA и MPS
