"""Упражнение 2. MLP и цикл обучения на PyTorch.

Теория: разделы «Многослойный перцептрон» и «PyTorch». Урок 2.
"""

import torch
from torch import nn
from torch.utils.data import DataLoader


def make_mlp(sizes: list[int], dropout: float = 0.0) -> nn.Sequential:
    """MLP по списку размеров [вход, скрытые…, выход].

    Между линейными слоями — ReLU, после ReLU — Dropout(dropout), если dropout > 0.
    После последнего линейного слоя ничего нет: модель выдаёт логиты.
    """
    raise NotImplementedError("TODO")


def count_parameters(model: nn.Module) -> int:
    """Число обучаемых параметров (requires_grad=True)."""
    raise NotImplementedError("TODO")


def train_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, device: str = "cpu") -> float:
    """Одна эпоха обучения с кросс-энтропией. Возвращает средние потери по всем примерам
    (потери батча взвешиваются его размером).
    """
    raise NotImplementedError("TODO: model.train(); для каждого батча — zero_grad, прямой проход, backward, step")


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str = "cpu") -> float:
    """Accuracy модели на данных `loader` в режиме оценки (dropout выключен)."""
    raise NotImplementedError("TODO")
