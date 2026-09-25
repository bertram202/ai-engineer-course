"""Настройки моделей курса.

Всё читается из переменных окружения и файла `.env` в корне репозитория
(ищется вверх от текущей папки). Код уроков не знает, где живёт модель:
локальная Ollama, свой сервер на llama.cpp/vLLM или облачный провайдер —
главное, чтобы API был OpenAI-совместимым.

Роли моделей:
    main  — основная модель (LLM_BASE_URL / LLM_API_KEY / LLM_MODEL);
            если LLM_BASE_URL не задан, main = local.
    local — локальная модель в Ollama (OLLAMA_BASE_URL / LLM_LOCAL_MODEL).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

from dotenv import find_dotenv, load_dotenv

Role = Literal["main", "local"]
CacheMode = Literal["off", "on", "record", "refresh", "replay"]
CACHE_MODES: tuple[str, ...] = ("off", "on", "record", "refresh", "replay")


def in_docker() -> bool:
    """Запущены ли мы внутри контейнера."""
    return Path("/.dockerenv").exists() or os.environ.get("MLCOURSE_IN_DOCKER") == "1"


def _host_url(url: str) -> str:
    """В контейнере `localhost` — это сам контейнер, а Ollama работает на хосте."""
    if not in_docker():
        return url
    for local in ("://localhost", "://127.0.0.1"):
        url = url.replace(local, "://host.docker.internal")
    return url


LOCAL_HOSTS = {"localhost", "127.0.0.1", "host.docker.internal"}


def safe_url(url: str) -> str:
    """Адрес для показа: локальный — как есть, удалённый — со скрытым хостом."""
    if "://" not in url:
        return url
    scheme, rest = url.split("://", 1)
    host, _, path = rest.partition("/")
    if host.split(":")[0] in LOCAL_HOSTS:
        return url
    return f"{scheme}://<удалённый сервер>/{path}"


def repo_root(start: Path | None = None) -> Path:
    """Корень репозитория курса (папка с _quarto.yml)."""
    here = (start or Path.cwd()).resolve()
    for d in (here, *here.parents):
        if (d / "_quarto.yml").exists():
            return d
    raise FileNotFoundError("Не нашёл корень курса (_quarto.yml) выше текущей папки")


@dataclass(frozen=True)
class Endpoint:
    """Куда и с какой моделью ходить для конкретной роли."""

    role: str
    base_url: str
    api_key: str
    model: str

    def __repr__(self) -> str:  # ключ не должен попадать в вывод ноутбуков
        return f"Endpoint(role={self.role!r}, model={self.model!r})"


@dataclass(frozen=True)
class Settings:
    main: Endpoint
    local: Endpoint
    embed: Endpoint
    cache_mode: str
    cache_dir: Path

    def endpoint(self, role: str) -> Endpoint:
        if role not in ("main", "local", "embed"):
            raise ValueError(f"Неизвестная роль модели: {role!r} (ожидается main, local или embed)")
        return getattr(self, role)

    def describe(self) -> str:
        """Таблица настроек для вывода в ноутбуке: без ключей и с замаскированным адресом удалённого сервера."""
        rows = [(ep.role, ep.model, safe_url(ep.base_url)) for ep in (self.main, self.local, self.embed)]
        rows.append(("кэш", self.cache_mode, "LLM_CACHE"))
        width = max(len(r[1]) for r in rows)
        return "\n".join(f"{role:<6} {model:<{width}}  {url}" for role, model, url in rows)


def load_settings() -> Settings:
    """Собирает настройки из окружения (значения из `.env` не перетирают уже заданные)."""
    load_dotenv(find_dotenv(usecwd=True), override=False)
    env = os.environ

    ollama = _host_url(env.get("OLLAMA_BASE_URL", "http://localhost:11434")).rstrip("/")
    local = Endpoint("local", f"{ollama}/v1", "ollama", env.get("LLM_LOCAL_MODEL", "qwen3:8b"))

    if env.get("LLM_BASE_URL"):
        main = Endpoint(
            "main",
            _host_url(env["LLM_BASE_URL"]).rstrip("/"),
            env.get("LLM_API_KEY", "none"),
            env.get("LLM_MODEL", local.model),
        )
    else:
        main = replace(local, role="main")

    embed = Endpoint("embed", local.base_url, local.api_key, env.get("EMBED_MODEL", "bge-m3"))

    cache_mode = env.get("LLM_CACHE", "on").lower()
    if cache_mode not in CACHE_MODES:
        raise ValueError(f"LLM_CACHE={cache_mode!r}: допустимо {', '.join(CACHE_MODES)}")
    cache_dir = Path(env.get("LLM_CACHE_DIR", Path.home() / ".cache" / "mlcourse" / "llm"))

    return Settings(main=main, local=local, embed=embed, cache_mode=cache_mode, cache_dir=cache_dir)
