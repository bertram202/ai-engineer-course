"""Упражнение 4. Собираем запрос к /chat/completions руками.

Теория: разделы «Анатомия запроса» и «Рассуждающие модели».
"""

from typing import Any


def build_chat_request(
    model: str,
    user: str,
    *,
    system: str | None = None,
    history: list[dict[str, str]] | None = None,
    thinking: bool = False,
    json_schema: dict[str, Any] | None = None,
    max_tokens: int | None = None,
) -> dict[str, Any]:
    """Тело запроса в формате OpenAI Chat Completions.

    Правила:
    * `messages`: сначала system (если задан), затем history (как есть), затем сообщение user;
    * если `thinking` выключен — добавить параметры, выключающие рассуждения
      и в Ollama (`reasoning_effort: "none"`), и в llama.cpp
      (`chat_template_kwargs: {"enable_thinking": False}`);
    * если задана `json_schema` — `response_format` вида
      `{"type": "json_schema", "json_schema": {"name": "response", "schema": <схема>}}`;
    * `max_tokens` добавлять, только если он задан;
    * исходный список `history` менять нельзя.
    """
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.extend(history or [])
    messages.append({"role": "user", "content": user})

    request: dict[str, Any] = {"model": model, "messages": messages}
    if not thinking:
        request["reasoning_effort"] = "none"
        request["chat_template_kwargs"] = {"enable_thinking": False}
    if json_schema is not None:
        request["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "response", "schema": json_schema},
        }
    if max_tokens is not None:
        request["max_tokens"] = max_tokens
    return request
