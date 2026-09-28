"""Рисунки к теории M03. Запуск из папки модуля: `uv run python theory/figures/make_figures.py`."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Rectangle
from torchvision import datasets

from mlcourse import data_dir, repo_root

OUT = Path(__file__).parent
plt.style.use(repo_root() / "quarto" / "figures.mplstyle")


def activations() -> None:
    """Функции активации и их производные."""
    x = torch.linspace(-4, 4, 400, requires_grad=True)
    funcs = {
        "sigmoid": torch.sigmoid,
        "tanh": torch.tanh,
        "ReLU": torch.relu,
        "GELU": torch.nn.functional.gelu,
        "SiLU (Swish)": torch.nn.functional.silu,
    }
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.2))
    for name, f in funcs.items():
        y = f(x)
        (g,) = torch.autograd.grad(y.sum(), x)
        a1.plot(x.detach(), y.detach(), label=name)
        a2.plot(x.detach(), g, label=name)
    a1.set(title="функция", ylim=(-1.5, 4))
    a2.set(title="производная", ylim=(-0.2, 1.2))
    a1.legend(fontsize=8)
    fig.savefig(OUT / "activations.png")
    plt.close(fig)


def init_variance() -> None:
    """Стандартное отклонение активаций по слоям глубокой сети при разной инициализации."""
    torch.manual_seed(0)
    d, depth = 512, 30
    x0 = torch.randn(1000, d)
    cases = {
        "N(0, 0.01²), tanh": (lambda: torch.randn(d, d) * 0.01, torch.tanh),
        "N(0, 1), tanh": (lambda: torch.randn(d, d), torch.tanh),
        "Xavier, tanh": (lambda: torch.randn(d, d) / d**0.5, torch.tanh),
        "Kaiming, ReLU": (lambda: torch.randn(d, d) * (2 / d) ** 0.5, torch.relu),
    }
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 3.4))
    for name, (make_w, act) in cases.items():
        h, stds, saturated = x0, [], []
        for _ in range(depth):
            h = act(h @ make_w())
            stds.append(h.std().item())
            saturated.append((h.abs() > 0.97).float().mean().item() if act is torch.tanh else 0.0)
        ax.semilogy(range(1, depth + 1), stds, marker="o", ms=2.5, label=name)
        ax2.plot(range(1, depth + 1), saturated, marker="o", ms=2.5, label=name)
    ax.set(xlabel="номер слоя", ylabel="std активаций", title="Разброс активаций по слоям")
    ax2.set(xlabel="номер слоя", ylabel="доля", title="Насыщенные нейроны tanh (|h| > 0,97)")
    ax.legend(fontsize=8)
    fig.savefig(OUT / "init_variance.png")
    plt.close(fig)


def convolution() -> None:
    """Свёртка 3×3 по входу 6×6 без паддинга: одно положение ядра и соответствующий выход."""
    rng = np.random.default_rng(3)
    inp = rng.integers(0, 4, size=(6, 6))
    ker = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
    out = np.array([[(inp[i : i + 3, j : j + 3] * ker).sum() for j in range(4)] for i in range(4)])

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.4), gridspec_kw={"width_ratios": [6, 3, 4]})
    for ax, mat, title in zip(axes, [inp, ker, out], ["вход 6×6", "ядро 3×3", "выход 4×4"]):
        ax.imshow(np.zeros_like(mat), cmap="Greys", vmin=0, vmax=1)
        for (i, j), v in np.ndenumerate(mat):
            ax.text(j, i, str(v), ha="center", va="center", fontsize=11)
        ax.set_xticks(np.arange(-0.5, mat.shape[1]), minor=True)
        ax.set_yticks(np.arange(-0.5, mat.shape[0]), minor=True)
        ax.grid(which="minor", color="0.6", lw=0.8)
        ax.grid(which="major", visible=False)
        ax.tick_params(which="both", length=0, labelbottom=False, labelleft=False)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(title)
    axes[0].add_patch(Rectangle((0.5, 0.5), 3, 3, fill=False, ec="#E53935", lw=3))
    axes[2].add_patch(Rectangle((0.5, 0.5), 1, 1, fill=False, ec="#E53935", lw=3))
    fig.savefig(OUT / "convolution.png")
    plt.close(fig)


def vit_patches() -> None:
    """Спутниковый снимок EuroSAT 64×64, разрезанный на 16 патчей 16×16 — «токены» ViT."""
    ds = datasets.EuroSAT(data_dir("eurosat"), download=True)
    img = np.asarray(ds[ds.targets.index(8) + 5][0])  # класс River
    fig = plt.figure(figsize=(9, 3.2))
    ax = fig.add_axes((0.0, 0.05, 0.3, 0.85))
    ax.imshow(img)
    for k in range(1, 4):
        ax.axhline(16 * k - 0.5, color="w", lw=1.5)
        ax.axvline(16 * k - 0.5, color="w", lw=1.5)
    ax.set_title("снимок 64×64")
    ax.axis("off")
    for n in range(16):
        i, j = divmod(n, 4)
        sub = fig.add_axes((0.36 + n * 0.04, 0.35, 0.036, 0.3))
        sub.imshow(img[16 * i : 16 * i + 16, 16 * j : 16 * j + 16])
        sub.axis("off")
    fig.text(0.68, 0.18, "16 патчей → 16 векторов-«токенов» → трансформер", ha="center", fontsize=11)
    fig.savefig(OUT / "vit_patches.png")
    plt.close(fig)


def contrastive() -> None:
    """Матрица сходства в пакете: на диагонали — положительные пары, остальное — отрицательные."""
    rng = np.random.default_rng(0)
    n = 8
    sim = rng.normal(0.1, 0.12, (n, n))
    sim[np.diag_indices(n)] = rng.normal(0.8, 0.06, n)
    fig, ax = plt.subplots(figsize=(4.4, 3.8))
    im = ax.imshow(sim, cmap="viridis", vmin=-0.2, vmax=1)
    ax.set_xticks(range(n), [f"$b_{i + 1}$" for i in range(n)])
    ax.set_yticks(range(n), [f"$a_{i + 1}$" for i in range(n)])
    ax.grid(False)
    ax.set_title("cos(aᵢ, bⱼ): строка — задача классификации")
    fig.colorbar(im, fraction=0.046)
    fig.savefig(OUT / "contrastive.png")
    plt.close(fig)


if __name__ == "__main__":
    for make in (activations, init_variance, convolution, vit_patches, contrastive):
        make()
        print("✓", make.__name__)
