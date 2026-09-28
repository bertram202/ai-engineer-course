"""Общая библиотека курса «AI-инженер: от устройства LLM до мультиагентных систем»."""

from .config import in_docker, load_settings, repo_root, safe_url
from .data import data_dir, fetch
from .dl import get_device, seed_everything
from .llm import (
    cassette_dir,
    chat,
    embed,
    get_async_client,
    get_chat_model,
    get_client,
    get_http_client,
    get_ollama_http_client,
    model_name,
    no_thinking,
    reasoning_text,
    set_cassette_dir,
)


def describe_settings() -> str:
    """Какие модели настроены (без ключей и с замаскированным удалённым адресом)."""
    return load_settings().describe()


__all__ = [
    "cassette_dir",
    "chat",
    "data_dir",
    "describe_settings",
    "embed",
    "fetch",
    "get_async_client",
    "get_chat_model",
    "get_client",
    "get_device",
    "get_http_client",
    "get_ollama_http_client",
    "in_docker",
    "load_settings",
    "model_name",
    "no_thinking",
    "reasoning_text",
    "repo_root",
    "safe_url",
    "seed_everything",
    "set_cassette_dir",
]

__version__ = "0.1.0"
