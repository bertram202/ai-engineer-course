"""Упражнение 3. Классификатор тональности на LLM.

Теория: раздел «Дообучение или промпт». Урок 3.
Функции принимают OpenAI-совместимый клиент: в уроке это `mlcourse.get_client(...)`,
в тестах — фейковая модель из `mlcourse.testing`.
"""

import json

from openai import OpenAI

LABELS = ("negative", "neutral", "positive")


def build_messages(text: str, system: str, examples: list[tuple[str, str]] | None = None) -> list[dict]:
    """Сообщения для chat/completions: system, затем пары примеров (user: текст, assistant: {"label": …}),
    затем сам отзыв от user. Ответы-примеры — JSON без экранирования кириллицы.
    """
    raise NotImplementedError("TODO")


def parse_label(content: str) -> str:
    """Метка из ответа модели. Сначала пробуем JSON {"label": …}; если это не JSON, ищем в тексте
    ровно одну из меток LABELS (без учёта регистра). Если метку определить нельзя — ValueError.
    """
    raise NotImplementedError("TODO: json.JSONDecodeError — сигнал перейти к поиску метки в тексте")


def stars_to_label(stars: int) -> str:
    """Как размечен RuReviews: 1–2 звезды → negative, 3 → neutral, 4–5 → positive."""
    raise NotImplementedError("TODO")


def classify(client: OpenAI, model: str, text: str, system: str, examples: list[tuple[str, str]] | None = None) -> str:
    """Спросить модель (temperature=0, max_tokens=20) и вернуть метку."""
    raise NotImplementedError("TODO")


def per_class_recall(y_true: list[str], y_pred: list[str]) -> dict[str, float]:
    """Recall каждого класса из LABELS: доля его примеров, предсказанных верно (0.0, если примеров нет)."""
    raise NotImplementedError("TODO")
