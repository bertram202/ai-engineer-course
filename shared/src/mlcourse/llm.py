"""Клиенты моделей курса: openai SDK, «голый» httpx и LangChain — все с кассетами.

    from mlcourse.llm import get_client, model_name, no_thinking

    client = get_client()                       # роль main из .env
    reply = client.chat.completions.create(
        model=model_name(),
        messages=[{"role": "user", "content": "Привет!"}],
        extra_body=no_thinking(),               # выключить режим рассуждений
    )
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import httpx
import numpy as np
from openai import AsyncOpenAI, DefaultAsyncHttpxClient, DefaultHttpxClient, OpenAI

from . import cassette as _cas
from .cassette import CassetteStore, CassetteTransport
from .config import Settings, load_settings

# openai SDK 3.x построен на httpx2; для старых версий — классический httpx
_SDK_HTTPX = _cas.httpx2 or httpx

_cassette_dir: Path | None = None


def set_cassette_dir(path: str | os.PathLike[str]) -> None:
    """Куда писать и откуда читать кассеты (по умолчанию `./cassettes` от текущей папки)."""
    global _cassette_dir
    _cassette_dir = Path(path).resolve()


def cassette_dir() -> Path:
    if _cassette_dir is not None:
        return _cassette_dir
    return Path(os.environ.get("LLM_CASSETTE_DIR", "cassettes")).resolve()


def _store(settings: Settings) -> CassetteStore:
    return CassetteStore(cassette_dir(), settings.cache_dir, settings.cache_mode)


def _sdk_transports(role: str, settings: Settings) -> tuple[Any, Any]:
    """Транспорты с кассетами для openai SDK (и всего, что поверх него)."""
    ep = settings.endpoint(role)
    store, base_path = _store(settings), httpx.URL(ep.base_url).path
    sync = _cas.SDKCassetteTransport(_SDK_HTTPX.HTTPTransport(), store, role=role, base_path=base_path)
    async_ = _cas.AsyncSDKCassetteTransport(_SDK_HTTPX.AsyncHTTPTransport(), store, role=role, base_path=base_path)
    return sync, async_


def _retries(settings: Settings, max_retries: int | None) -> int:
    if settings.cache_mode == "replay":
        return 0
    return 2 if max_retries is None else max_retries


def model_name(role: str = "main") -> str:
    """Имя модели для роли: main, local или embed."""
    return load_settings().endpoint(role).model


def get_client(role: str = "main", *, timeout: float = 600.0, max_retries: int | None = None) -> OpenAI:
    """Синхронный клиент openai SDK для выбранной роли."""
    s = load_settings()
    ep = s.endpoint(role)
    sync, _ = _sdk_transports(role, s)
    return OpenAI(
        base_url=ep.base_url,
        api_key=ep.api_key,
        timeout=timeout,
        max_retries=_retries(s, max_retries),
        http_client=DefaultHttpxClient(transport=sync),
    )


def get_async_client(role: str = "main", *, timeout: float = 600.0, max_retries: int | None = None) -> AsyncOpenAI:
    """Асинхронный клиент openai SDK для выбранной роли."""
    s = load_settings()
    ep = s.endpoint(role)
    _, async_ = _sdk_transports(role, s)
    return AsyncOpenAI(
        base_url=ep.base_url,
        api_key=ep.api_key,
        timeout=timeout,
        max_retries=_retries(s, max_retries),
        http_client=DefaultAsyncHttpxClient(transport=async_),
    )


def get_http_client(role: str = "main", *, timeout: float = 600.0) -> httpx.Client:
    """«Голый» httpx-клиент с base_url и ключом: чтобы смотреть на API без SDK.

    Пути запросов — относительные: `client.post("chat/completions", json=...)`.
    """
    s = load_settings()
    ep = s.endpoint(role)
    transport = CassetteTransport(
        httpx.HTTPTransport(), _store(s), role=role, base_path=httpx.URL(ep.base_url).path
    )
    return httpx.Client(
        base_url=ep.base_url + "/",
        headers={"Authorization": f"Bearer {ep.api_key}"},
        transport=transport,
        timeout=timeout,
    )


def get_ollama_http_client(*, timeout: float = 600.0) -> httpx.Client:
    """httpx-клиент к нативному API Ollama (`api/tags`, `api/ps`, …) — с кассетами."""
    s = load_settings()
    root = s.local.base_url.removesuffix("/v1")
    transport = CassetteTransport(httpx.HTTPTransport(), _store(s), role="ollama", base_path="")
    return httpx.Client(base_url=root + "/", transport=transport, timeout=timeout)


def no_thinking() -> dict[str, Any]:
    """Параметры, выключающие режим рассуждений у Qwen3-подобных моделей.

    Ollama понимает `reasoning_effort`, llama.cpp — `chat_template_kwargs`;
    лишний параметр каждый сервер просто игнорирует.
    """
    return {"reasoning_effort": "none", "chat_template_kwargs": {"enable_thinking": False}}


def reasoning_text(message: Any) -> str:
    """Текст рассуждений из ответа (llama.cpp: `reasoning_content`, Ollama: `reasoning`)."""
    data = message if isinstance(message, dict) else (getattr(message, "model_extra", None) or {})
    return data.get("reasoning_content") or data.get("reasoning") or ""


def chat(
    prompt: str | Sequence[dict[str, Any]],
    *,
    role: str = "main",
    system: str | None = None,
    thinking: bool = False,
    **kwargs: Any,
) -> str:
    """Короткий способ получить текстовый ответ модели."""
    messages = [{"role": "user", "content": prompt}] if isinstance(prompt, str) else list(prompt)
    if system:
        messages.insert(0, {"role": "system", "content": system})
    extra_body = {**(kwargs.pop("extra_body", None) or {}), **({} if thinking else no_thinking())}
    resp = get_client(role).chat.completions.create(
        model=model_name(role), messages=messages, extra_body=extra_body, **kwargs
    )
    return resp.choices[0].message.content or ""


def embed(texts: str | Sequence[str], *, model: str | None = None) -> np.ndarray:
    """Эмбеддинги текстов (роль embed, по умолчанию bge-m3 в Ollama): массив [n, dim]."""
    items = [texts] if isinstance(texts, str) else list(texts)
    resp = get_client("embed").embeddings.create(model=model or model_name("embed"), input=items)
    return np.array([d.embedding for d in sorted(resp.data, key=lambda d: d.index)], dtype=np.float32)


def get_chat_model(role: str = "main", *, thinking: bool = False, **kwargs: Any):
    """Модель LangChain (`ChatOpenAI`) для роли — с кассетами. Нужен пакет langchain-openai."""
    from langchain_openai import ChatOpenAI

    s = load_settings()
    ep = s.endpoint(role)
    sync, async_ = _sdk_transports(role, s)
    extra_body = {**(kwargs.pop("extra_body", None) or {}), **({} if thinking else no_thinking())}
    return ChatOpenAI(
        model=ep.model,
        base_url=ep.base_url,
        api_key=ep.api_key,
        max_retries=_retries(s, kwargs.pop("max_retries", None)),
        http_client=DefaultHttpxClient(transport=sync),
        http_async_client=DefaultAsyncHttpxClient(transport=async_),
        extra_body=extra_body or None,
        **kwargs,
    )
