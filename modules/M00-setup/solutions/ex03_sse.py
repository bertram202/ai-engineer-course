"""Упражнение 3. Разбор потокового ответа (Server-Sent Events).

Сервер присылает строки вида:

    data: {"choices":[{"index":0,"delta":{"content":"При"}}]}
    <пустая строка>
    data: [DONE]

Теория: раздел «Потоковая выдача (streaming)».
"""

import json
from collections.abc import Iterable, Iterator
from typing import Any


def parse_sse(lines: Iterable[str]) -> Iterator[dict[str, Any]]:
    """Превращает строки SSE-потока в JSON-события.

    * интересны только строки, начинающиеся с `data:` (пробел после двоеточия может быть, а может не быть);
    * пустые строки и комментарии (строки, начинающиеся с `:`) пропускаются;
    * на `data: [DONE]` разбор заканчивается, даже если дальше что-то есть.
    """
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith(":") or not line.startswith("data:"):
            continue
        payload = line[len("data:") :].strip()
        if payload == "[DONE]":
            return
        yield json.loads(payload)


def collect_stream(events: Iterable[dict[str, Any]]) -> tuple[str, str]:
    """Собирает из событий итоговые (ответ, рассуждения).

    * текст ответа — в `choices[0].delta.content`;
    * рассуждения — в `choices[0].delta.reasoning_content` (llama.cpp) или `...reasoning` (Ollama);
    * у последнего события со статистикой `choices` бывает пустым списком;
    * поля могут отсутствовать или быть равны None.
    """
    content, reasoning = [], []
    for event in events:
        choices = event.get("choices") or []
        if not choices:
            continue
        delta = choices[0].get("delta") or {}
        content.append(delta.get("content") or "")
        reasoning.append(delta.get("reasoning_content") or delta.get("reasoning") or "")
    return "".join(content), "".join(reasoning)
