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
    messages = [{"role": "system", "content": system}]
    for example_text, label in examples or []:
        messages.append({"role": "user", "content": example_text})
        messages.append({"role": "assistant", "content": json.dumps({"label": label}, ensure_ascii=False)})
    messages.append({"role": "user", "content": text})
    return messages


def parse_label(content: str) -> str:
    """Метка из ответа модели. Сначала пробуем JSON {"label": …}; если это не JSON, ищем в тексте
    ровно одну из меток LABELS (без учёта регистра). Если метку определить нельзя — ValueError.
    """
    # подсказка: json.JSONDecodeError — сигнал перейти к поиску метки в тексте
    try:
        label = json.loads(content)["label"]
    except (json.JSONDecodeError, KeyError, TypeError):
        found = [lab for lab in LABELS if lab in content.lower()]
        if len(found) != 1:
            raise ValueError(f"не удалось понять метку в ответе: {content!r}") from None
        label = found[0]
    if label not in LABELS:
        raise ValueError(f"неизвестная метка: {label!r}")
    return label


def stars_to_label(stars: int) -> str:
    """Как размечен RuReviews: 1–2 звезды → negative, 3 → neutral, 4–5 → positive."""
    if not 1 <= stars <= 5:
        raise ValueError(f"звёзд должно быть от 1 до 5, получено {stars}")
    return "negative" if stars <= 2 else "neutral" if stars == 3 else "positive"


def classify(client: OpenAI, model: str, text: str, system: str, examples: list[tuple[str, str]] | None = None) -> str:
    """Спросить модель (temperature=0, max_tokens=20) и вернуть метку."""
    response = client.chat.completions.create(
        model=model, messages=build_messages(text, system, examples), temperature=0, max_tokens=20
    )
    return parse_label(response.choices[0].message.content)


def per_class_recall(y_true: list[str], y_pred: list[str]) -> dict[str, float]:
    """Recall каждого класса из LABELS: доля его примеров, предсказанных верно (0.0, если примеров нет)."""
    result = {}
    for label in LABELS:
        idx = [i for i, y in enumerate(y_true) if y == label]
        result[label] = sum(y_pred[i] == label for i in idx) / len(idx) if idx else 0.0
    return result
