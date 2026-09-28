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
    layers: list[nn.Module] = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


def count_parameters(model: nn.Module) -> int:
    """Число обучаемых параметров (requires_grad=True)."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, device: str = "cpu") -> float:
    """Одна эпоха обучения с кросс-энтропией. Возвращает средние потери по всем примерам
    (потери батча взвешиваются его размером).
    """
    # подсказка: model.train(); для каждого батча — zero_grad, прямой проход, backward, step
    model.train()
    total, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = nn.functional.cross_entropy(model(x), y)
        loss.backward()
        optimizer.step()
        total += loss.item() * len(x)
        n += len(x)
    return total / n


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str = "cpu") -> float:
    """Accuracy модели на данных `loader` в режиме оценки (dropout выключен)."""
    model.eval()
    correct, n = 0, 0
    for x, y in loader:
        pred = model(x.to(device)).argmax(dim=1).cpu()
        correct += int((pred == y).sum())
        n += len(y)
    return correct / n
