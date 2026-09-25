"""Проверка моделей курса: `make llm-check` или `python -m mlcourse.check`.

Для каждой роли (main, local) проверяет: доступность, чат, стриминг и скорость,
tool calling, JSON Schema; для embed — эмбеддинги. Кэш при проверке выключен.
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass

import httpx

from .config import load_settings
from .llm import get_client, no_thinking

RECOMMENDED_CTX = 16384


@dataclass
class Result:
    name: str
    ok: bool
    detail: str


def _run(name: str, fn: Callable[[], str]) -> Result:
    try:
        return Result(name, True, fn())
    except Exception as e:  # noqa: BLE001 — показываем любую ошибку как провал проверки
        return Result(name, False, f"{type(e).__name__}: {str(e)[:160]}")


def check_chat_role(role: str) -> list[Result]:
    ep = load_settings().endpoint(role)
    client = get_client(role, timeout=300, max_retries=0)
    results: list[Result] = []

    def models() -> str:
        ids = [m.id for m in client.models.list().data]
        if ep.model not in ids:
            raise RuntimeError(f"модели {ep.model!r} нет на сервере; доступны: {', '.join(ids[:8])}")
        return f"{ep.model} найдена"

    def stream_speed() -> str:
        t0, ttft, pieces, usage = time.perf_counter(), None, 0, None
        stream = client.chat.completions.create(
            model=ep.model,
            messages=[{"role": "user", "content": "Перечисли через запятую 20 русских городов."}],
            stream=True,
            stream_options={"include_usage": True},
            max_tokens=200,
            extra_body=no_thinking(),
        )
        for chunk in stream:
            if chunk.usage:
                usage = chunk.usage
            if chunk.choices and chunk.choices[0].delta.content:
                ttft = ttft or time.perf_counter() - t0
                pieces += 1
        total = time.perf_counter() - t0
        tokens = usage.completion_tokens if usage else pieces
        speed = tokens / max(total - (ttft or 0), 1e-6)
        return f"TTFT {ttft or 0:.2f} с, {tokens} ток. за {total:.1f} с ≈ {speed:.1f} ток/с"

    def tools() -> str:
        resp = client.chat.completions.create(
            model=ep.model,
            messages=[{"role": "user", "content": "Какая погода в Казани?"}],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Текущая погода в городе",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                            "required": ["city"],
                        },
                    },
                }
            ],
            extra_body=no_thinking(),
        )
        calls = resp.choices[0].message.tool_calls or []
        if not calls or calls[0].function.name != "get_weather":
            raise RuntimeError("модель не вызвала инструмент")
        return f"get_weather({calls[0].function.arguments})"

    def json_schema() -> str:
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
            "required": ["name", "age"],
        }
        resp = client.chat.completions.create(
            model=ep.model,
            messages=[{"role": "user", "content": "Извлеки данные: Анна Смирнова, 29 лет."}],
            response_format={"type": "json_schema", "json_schema": {"name": "person", "schema": schema}},
            extra_body=no_thinking(),
        )
        data = json.loads(resp.choices[0].message.content or "")
        if not isinstance(data.get("age"), int):
            raise RuntimeError(f"ответ не соответствует схеме: {data}")
        return json.dumps(data, ensure_ascii=False)

    results.append(_run("сервер и модель", models))
    if not results[-1].ok:
        return results
    results += [
        _run("стриминг и скорость", stream_speed),
        _run("tool calling", tools),
        _run("JSON Schema", json_schema),
    ]
    if role == "local" or ep.base_url.startswith(load_settings().local.base_url):
        results.append(_run("контекст Ollama", lambda: _ollama_context(ep.model)))
    return results


def _ollama_context(model: str) -> str:
    base = load_settings().local.base_url.removesuffix("/v1")
    loaded = httpx.get(f"{base}/api/ps", timeout=10).json().get("models", [])
    ctx = next((m.get("context_length") for m in loaded if m.get("name", "").startswith(model)), None)
    if ctx is None:
        return "модель не загружена в память — пропуск"
    if ctx < RECOMMENDED_CTX:
        raise RuntimeError(
            f"контекст {ctx} < {RECOMMENDED_CTX}: перезапустите Ollama с OLLAMA_CONTEXT_LENGTH={RECOMMENDED_CTX}"
        )
    return f"контекст {ctx}"


def check_embed() -> list[Result]:
    def run() -> str:
        from .llm import embed

        vecs = embed(["кошка сидит на окне", "a cat is sitting on the window", "курс доллара вырос"])
        vecs /= (vecs**2).sum(axis=1, keepdims=True) ** 0.5
        same, other = float(vecs[0] @ vecs[1]), float(vecs[0] @ vecs[2])
        if same <= other:
            raise RuntimeError(f"похожие тексты не ближе непохожих: {same:.2f} ≤ {other:.2f}")
        return f"dim={vecs.shape[1]}, cos(похожие)={same:.2f} > cos(разные)={other:.2f}"

    return [_run("эмбеддинги", run)]


def main(argv: list[str] | None = None) -> int:
    os.environ["LLM_CACHE"] = "off"
    roles = (argv if argv is not None else sys.argv[1:]) or ["main", "local", "embed"]
    s = load_settings()
    if "main" in roles and "local" in roles and s.main.base_url == s.local.base_url and s.main.model == s.local.model:
        roles = [r for r in roles if r != "main"]
        print("main = local (LLM_BASE_URL не задан) — проверяю одну модель\n")

    failed = 0
    for role in roles:
        ep = s.endpoint(role)
        print(f"── {role}: {ep.model}")
        for r in check_embed() if role == "embed" else check_chat_role(role):
            failed += not r.ok
            print(f"   {'✅' if r.ok else '❌'} {r.name:<22} {r.detail}")
        print()
    print("Всё в порядке 🎉" if not failed else f"Проблем: {failed}. Подсказки — в modules/M00-setup/README.md")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
