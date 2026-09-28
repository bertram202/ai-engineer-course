"""Упражнение 2. SVD, низкоранговые приближения и LoRA.

Теория: раздел «SVD и низкоранговые приближения». Урок 1.
"""

import numpy as np


def low_rank_approx(a: np.ndarray, k: int) -> np.ndarray:
    """Лучшее приближение матрицы `a` матрицей ранга `k` (усечённое SVD)."""
    raise NotImplementedError("TODO")


def approx_error(singular_values: np.ndarray, k: int) -> float:
    """Ошибка ‖A − A_k‖_F лучшего приближения ранга `k` — по сингулярным числам, без самой матрицы."""
    raise NotImplementedError("TODO")


def rank_for_energy(singular_values: np.ndarray, share: float) -> int:
    """Наименьший ранг k, при котором Σ_{i≤k} σᵢ² составляет не меньше `share` от Σ σᵢ²."""
    raise NotImplementedError("TODO")


def lora_params(d_in: int, d_out: int, r: int) -> int:
    """Число параметров LoRA-адаптера ранга `r` для матрицы d_in → d_out (матрицы A и B)."""
    raise NotImplementedError("TODO")


def lora_share(shapes: list[tuple[int, int]], r: int) -> float:
    """Доля параметров LoRA ранга `r` от полных матриц заданных форм [(d_in, d_out), …]."""
    raise NotImplementedError("TODO")
