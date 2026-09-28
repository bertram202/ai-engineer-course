"""Упражнение 2. Токенизатор BPE с нуля.

Теория: раздел «Токенизация». Урок 2.
Слово — кортеж символов с маркером конца слова `</w>`: «низ» → ('н', 'и', 'з', '</w>').
Корпус — словарь {слово-кортеж: сколько раз встречается}.
"""

from collections import Counter

END = "</w>"


def word_to_symbols(word: str) -> tuple[str, ...]:  # готово
    return (*word, END)


def pair_counts(corpus: dict[tuple[str, ...], int]) -> Counter:
    """Сколько раз каждая пара соседних символов встречается в корпусе с учётом частот слов."""
    counts: Counter = Counter()
    for symbols, freq in corpus.items():
        for pair in zip(symbols, symbols[1:]):
            counts[pair] += freq
    return counts


def merge_pair(corpus: dict[tuple[str, ...], int], pair: tuple[str, str]) -> dict[tuple[str, ...], int]:
    """Новый корпус, в котором каждое вхождение пары (a, b) слито в один символ a+b (слева направо)."""
    merged = {}
    for symbols, freq in corpus.items():
        out, i = [], 0
        while i < len(symbols):
            if i + 1 < len(symbols) and (symbols[i], symbols[i + 1]) == pair:
                out.append(symbols[i] + symbols[i + 1])
                i += 2
            else:
                out.append(symbols[i])
                i += 1
        merged[tuple(out)] = merged.get(tuple(out), 0) + freq
    return merged


def train_bpe(word_counts: dict[str, int], num_merges: int) -> list[tuple[str, str]]:
    """Список слияний в порядке обучения. На каждом шаге сливается самая частая пара;
    при равных частотах — наименьшая по алфавиту. Если пар не осталось, обучение останавливается.
    """
    # подсказка: min(counts, key=lambda p: (-counts[p], p)) выбирает частую пару с учётом ничьих
    corpus = {word_to_symbols(w): c for w, c in word_counts.items()}
    merges = []
    for _ in range(num_merges):
        counts = pair_counts(corpus)
        if not counts:
            break
        best = min(counts, key=lambda p: (-counts[p], p))
        corpus = merge_pair(corpus, best)
        merges.append(best)
    return merges


def apply_bpe(word: str, merges: list[tuple[str, str]]) -> list[str]:
    """Разбить слово на токены, применяя слияния в том порядке, в котором они выучены."""
    symbols = {word_to_symbols(word): 1}
    for pair in merges:
        symbols = merge_pair(symbols, pair)
    return list(next(iter(symbols)))
