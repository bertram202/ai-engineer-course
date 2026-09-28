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
    raise NotImplementedError("TODO")


def collect_stream(events: Iterable[dict[str, Any]]) -> tuple[str, str]:
    """Собирает из событий итоговые (ответ, рассуждения).

    * текст ответа — в `choices[0].delta.content`;
    * рассуждения — в `choices[0].delta.reasoning_content` (llama.cpp) или `...reasoning` (Ollama);
    * у последнего события со статистикой `choices` бывает пустым списком;
    * поля могут отсутствовать или быть равны None.
    """
    raise NotImplementedError("TODO")
