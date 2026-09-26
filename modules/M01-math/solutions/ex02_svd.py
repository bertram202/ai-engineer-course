"""Упражнение 2. SVD, низкоранговые приближения и LoRA.

Теория: раздел «SVD и низкоранговые приближения». Урок 1.
"""

import numpy as np


def low_rank_approx(a: np.ndarray, k: int) -> np.ndarray:
    """Лучшее приближение матрицы `a` матрицей ранга `k` (усечённое SVD)."""
    u, s, vt = np.linalg.svd(a, full_matrices=False)
    return (u[:, :k] * s[:k]) @ vt[:k]


def approx_error(singular_values: np.ndarray, k: int) -> float:
    """Ошибка ‖A − A_k‖_F лучшего приближения ранга `k` — по сингулярным числам, без самой матрицы."""
    return float(np.sqrt(np.sum(np.asarray(singular_values)[k:] ** 2)))


def rank_for_energy(singular_values: np.ndarray, share: float) -> int:
    """Наименьший ранг k, при котором Σ_{i≤k} σᵢ² составляет не меньше `share` от Σ σᵢ²."""
    s2 = np.sort(np.asarray(singular_values, dtype=float))[::-1] ** 2
    energy = np.cumsum(s2) / s2.sum()
    return int(np.searchsorted(energy, share - 1e-12) + 1)


def lora_params(d_in: int, d_out: int, r: int) -> int:
    """Число параметров LoRA-адаптера ранга `r` для матрицы d_in → d_out (матрицы A и B)."""
    return r * (d_in + d_out)


def lora_share(shapes: list[tuple[int, int]], r: int) -> float:
    """Доля параметров LoRA ранга `r` от полных матриц заданных форм [(d_in, d_out), …]."""
    full = sum(d_in * d_out for d_in, d_out in shapes)
    return sum(lora_params(d_in, d_out, r) for d_in, d_out in shapes) / full
