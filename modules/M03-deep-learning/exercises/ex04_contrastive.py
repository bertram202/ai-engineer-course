"""Упражнение 4. InfoNCE, zero-shot и recall@k.

Теория: раздел «Эмбеддинги как выучиваемые представления». Уроки 3 и 4.
"""

import torch
import torch.nn.functional as F


def info_nce(a: torch.Tensor, b: torch.Tensor, temperature: float = 0.05, symmetric: bool = False) -> torch.Tensor:
    """InfoNCE для батча пар (a_i, b_i), a и b: [B, d].

    Векторы сначала нормируются. Потери — кросс-энтропия по строкам матрицы a·bᵀ/τ,
    где правильный «класс» строки i — это i. При symmetric=True — среднее с тем же по столбцам.
    """
    raise NotImplementedError("TODO: F.normalize, затем F.cross_entropy(logits, torch.arange(B))")


def zero_shot_predict(image_emb: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
    """Номер класса для каждой картинки: текст класса с наибольшим косинусом.

    image_emb: [N, d], text_emb: [K, d] (по одному тексту на класс), векторы не нормированы.
    """
    raise NotImplementedError("TODO")


def recall_at_k(queries: torch.Tensor, docs: torch.Tensor, k: int) -> float:
    """Доля запросов i, для которых документ i входит в k самых похожих по косинусу."""
    raise NotImplementedError("TODO")
