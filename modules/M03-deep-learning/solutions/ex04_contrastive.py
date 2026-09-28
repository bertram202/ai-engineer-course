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
    # подсказка: F.normalize, затем F.cross_entropy(logits, torch.arange(B))
    a, b = F.normalize(a, dim=1), F.normalize(b, dim=1)
    logits = a @ b.T / temperature
    target = torch.arange(len(a), device=a.device)
    loss = F.cross_entropy(logits, target)
    if symmetric:
        loss = (loss + F.cross_entropy(logits.T, target)) / 2
    return loss


def zero_shot_predict(image_emb: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
    """Номер класса для каждой картинки: текст класса с наибольшим косинусом.

    image_emb: [N, d], text_emb: [K, d] (по одному тексту на класс), векторы не нормированы.
    """
    return (F.normalize(image_emb, dim=1) @ F.normalize(text_emb, dim=1).T).argmax(dim=1)


def recall_at_k(queries: torch.Tensor, docs: torch.Tensor, k: int) -> float:
    """Доля запросов i, для которых документ i входит в k самых похожих по косинусу."""
    sims = F.normalize(queries, dim=1) @ F.normalize(docs, dim=1).T
    top = sims.topk(k, dim=1).indices
    hits = (top == torch.arange(len(queries), device=top.device)[:, None]).any(dim=1)
    return hits.float().mean().item()
