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
    raise NotImplementedError("TODO")


def merge_pair(corpus: dict[tuple[str, ...], int], pair: tuple[str, str]) -> dict[tuple[str, ...], int]:
    """Новый корпус, в котором каждое вхождение пары (a, b) слито в один символ a+b (слева направо)."""
    raise NotImplementedError("TODO")


def train_bpe(word_counts: dict[str, int], num_merges: int) -> list[tuple[str, str]]:
    """Список слияний в порядке обучения. На каждом шаге сливается самая частая пара;
    при равных частотах — наименьшая по алфавиту. Если пар не осталось, обучение останавливается.
    """
    raise NotImplementedError("TODO: min(counts, key=lambda p: (-counts[p], p)) выбирает частую пару с учётом ничьих")


def apply_bpe(word: str, merges: list[tuple[str, str]]) -> list[str]:
    """Разбить слово на токены, применяя слияния в том порядке, в котором они выучены."""
    raise NotImplementedError("TODO")
