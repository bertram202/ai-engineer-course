"""Рисунки к теории M04. Запуск из папки модуля: `uv run python theory/figures/make_figures.py`."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoTokenizer

from mlcourse import repo_root

OUT = Path(__file__).parent
plt.style.use(repo_root() / "quarto" / "figures.mplstyle")


def timeline() -> None:
    """Хронология NLP: от мешка слов до открытых LLM."""
    events = [
        (1972, "TF-IDF"),
        (2003, "нейросетевая\nязыковая модель"),
        (2013, "word2vec"),
        (2014, "seq2seq на RNN"),
        (2015, "внимание\n(Bahdanau)"),
        (2017, "Transformer"),
        (2018, "BERT, GPT"),
        (2020, "GPT-3:\nfew-shot"),
        (2022, "ChatGPT\n(RLHF)"),
        (2024, "открытые LLM:\nLlama, Qwen"),
    ]
    fig, ax = plt.subplots(figsize=(11, 2.4))
    ax.axhline(0, color="0.6", lw=1.5, zorder=0)
    for i, (year, label) in enumerate(events):  # шкала по событиям, а не по годам: иначе 2013–2024 слипаются
        up = 1 if i % 2 == 0 else -1
        ax.plot([i, i], [0, 0.45 * up], color="0.7", lw=1)
        ax.scatter([i], [0], s=40, color="#3949AB", zorder=3)
        text = f"{year}\n{label}" if up > 0 else f"{label}\n{year}"
        ax.text(i, 0.52 * up, text, ha="center", va="bottom" if up > 0 else "top", fontsize=9)
    ax.set_xlim(-0.7, len(events) - 0.3)
    ax.set_ylim(-1.6, 1.6)
    ax.axis("off")
    fig.savefig(OUT / "timeline.png")
    plt.close(fig)


def attention_masks() -> None:
    """Кто на кого смотрит: encoder (все на всех), decoder (только назад), encoder-decoder (кросс-внимание)."""
    n = 6
    enc = np.ones((n, n))
    dec = np.tril(np.ones((n, n)))
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.3))
    titles = ["Encoder (BERT):\nкаждый токен видит все", "Decoder (GPT, Qwen):\nтолько себя и прошлое",
              "Encoder-decoder (T5), кросс-внимание:\nвыход видит весь вход"]
    for ax, mat, title in zip(axes, [enc, dec, np.ones((4, n))], titles):
        ax.imshow(mat, cmap="Blues", vmin=0, vmax=1.3)
        ax.set_title(title, fontsize=10)
        ax.set_xticks(range(mat.shape[1]), [f"x{j + 1}" for j in range(mat.shape[1])], fontsize=8)
        ax.set_yticks(range(mat.shape[0]), [f"{'y' if mat.shape[0] == 4 else 'x'}{i + 1}" for i in range(mat.shape[0])],
                      fontsize=8)
        ax.grid(False)
        ax.set_xticks(np.arange(-0.5, mat.shape[1]), minor=True)
        ax.set_yticks(np.arange(-0.5, mat.shape[0]), minor=True)
        ax.grid(which="minor", color="w", lw=1.5)
        ax.tick_params(which="minor", length=0)
        ax.set_xlabel("на какие токены смотрим", fontsize=8)
    axes[0].set_ylabel("какой токен считаем", fontsize=8)
    fig.savefig(OUT / "attention_masks.png")
    plt.close(fig)


def tokenizers() -> None:
    """Сколько токенов занимает один и тот же текст у разных токенизаторов."""
    texts = {
        "русский": "Платье пришло быстро, ткань приятная, но размер маломерит — пришлось вернуть. "
                   "Продавец вежливый, деньги вернули без вопросов.",
        "английский": "The dress arrived quickly, the fabric is nice, but it runs small, so I had to return it. "
                      "The seller was polite and refunded me without questions.",
    }
    names = {"openai-community/gpt2": "GPT-2", "cointegrated/rubert-tiny2": "rubert-tiny2", "Qwen/Qwen3-8B": "Qwen3"}
    counts = {label: [len(AutoTokenizer.from_pretrained(m)(t, add_special_tokens=False)["input_ids"])
                      for t in texts.values()] for m, label in names.items()}
    fig, ax = plt.subplots(figsize=(7, 3.2))
    x = np.arange(len(names))
    for k, lang in enumerate(texts):
        vals = [counts[label][k] for label in names.values()]
        bars = ax.bar(x + (k - 0.5) * 0.38, vals, width=0.38, label=lang)
        ax.bar_label(bars, fontsize=8)
    ax.set_xticks(x, list(names.values()))
    ax.set_ylabel("токенов")
    ax.set_title("Один и тот же отзыв в разных токенизаторах")
    ax.legend()
    fig.savefig(OUT / "tokenizers.png")
    plt.close(fig)
    print(counts)


if __name__ == "__main__":
    for make in (timeline, attention_masks, tokenizers):
        make()
        print("✓", make.__name__)
