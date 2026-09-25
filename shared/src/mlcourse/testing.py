"""Фейковый OpenAI-совместимый сервер для автотестов: без сети и без моделей.

    client, requests = fake_client(["Привет!", tool_call("get_weather", city="Казань")])
    client.chat.completions.create(model="fake", messages=[...])  # -> "Привет!"
    requests[0]["messages"]                                       # что ушло в «модель»
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

import httpx
from openai import OpenAI


def tool_call(name: str, call_id: str | None = None, **arguments: Any) -> dict[str, Any]:
    """Ответ модели, в котором она вызывает инструмент."""
    return {
        "tool_calls": [
            {
                "id": call_id or f"call_{name}",
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(arguments, ensure_ascii=False)},
            }
        ]
    }


def chat_completion(reply: str | dict[str, Any], *, model: str = "fake") -> dict[str, Any]:
    """JSON ответа /chat/completions в формате OpenAI."""
    message: dict[str, Any] = {"role": "assistant", "content": None}
    if isinstance(reply, str):
        message["content"] = reply
    else:
        message.update(reply)
    finish = "tool_calls" if message.get("tool_calls") else "stop"
    return {
        "id": "chatcmpl-fake",
        "object": "chat.completion",
        "created": 0,
        "model": model,
        "choices": [{"index": 0, "message": message, "finish_reason": finish}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


def fake_client(replies: Sequence[str | dict[str, Any]]) -> tuple[OpenAI, list[dict[str, Any]]]:
    """OpenAI-клиент, который по очереди отдаёт заготовленные ответы.

    Возвращает (клиент, список тел запросов) — по списку удобно проверять,
    что код отправил модели.
    """
    queue = list(replies)
    requests: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content or b"{}")
        requests.append(body)
        if not queue:
            return httpx.Response(500, json={"error": {"message": "fake_client: ответы закончились"}})
        return httpx.Response(200, json=chat_completion(queue.pop(0), model=body.get("model", "fake")))

    client = OpenAI(
        base_url="http://fake.local/v1",
        api_key="fake",
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    return client, requests
