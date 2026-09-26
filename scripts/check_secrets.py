#!/usr/bin/env python3
"""Проверка, что в файлы не попали секреты: `scripts/check_secrets.py [файлы…]`.

Без аргументов проверяет все файлы, отслеживаемые git. Ищет:

* значения из локального `.env`: ключи API и адрес удалённого сервера моделей
  (сами значения читаются из `.env` в момент проверки и нигде не хранятся);
* типичные форматы ключей (`sk-…`, `ghp_…` и т. п.);
* абсолютные пути с именем пользователя (`/Users/<имя>/`, `/home/<имя>/`) — в выводах ноутбуков.

Используется как pre-commit хук и в CI. Код возврата 1, если что-то найдено.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCAL_HOSTS = {"localhost", "127.0.0.1", "host.docker.internal", ""}
TRIVIAL_VALUES = {"", "ollama", "none", "x", "fake", "changeme"}

KEY_PATTERNS = [
    re.compile(r"\bsk-[A-Za-z0-9_-]{20,}"),  # OpenAI-подобные ключи
    re.compile(r"\bsk-ant-[A-Za-z0-9_-]{20,}"),  # Anthropic
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}"),  # GitHub
    re.compile(r"\bhf_[A-Za-z0-9]{30,}"),  # Hugging Face
]


def looks_like_key(token: str) -> bool:
    """Настоящие ключи случайны: в них есть и цифры, и буквы обоих регистров.
    Так отсекаются CSS-классы вроде `sk-toggleable__label` из HTML-вывода scikit-learn."""
    body = token.split("_", 1)[-1] if token.startswith(("gh", "hf_")) else token[3:]
    return any(c.isdigit() for c in body) and any(c.isupper() for c in body) and any(c.islower() for c in body)


HOME_PATH = re.compile(r"(/Users|/home)/(?!runner/)[A-Za-z0-9._-]+/")
SKIP_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".pdf", ".ttf", ".otf", ".woff", ".woff2", ".ico", ".lock"}


def env_secrets() -> list[str]:
    """Значения из .env, которые нельзя публиковать."""
    env_file = ROOT / ".env"
    if not env_file.exists():
        return []
    secrets = []
    for line in env_file.read_text(encoding="utf-8").splitlines():
        if "=" not in line or line.lstrip().startswith("#"):
            continue
        name, value = (part.strip().strip("'\"") for part in line.split("=", 1))
        if value.lower() in TRIVIAL_VALUES:
            continue
        if any(word in name for word in ("KEY", "TOKEN", "SECRET", "PASSWORD")):
            secrets.append(value)
        elif name.endswith("BASE_URL") and "://" in value:
            host = value.split("://", 1)[1].split("/", 1)[0].split(":")[0]
            if host not in LOCAL_HOSTS:
                secrets.append(host)
    return [s for s in secrets if len(s) >= 6]


def tracked_files() -> list[Path]:
    out = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, check=True).stdout
    return [ROOT / p for p in out.decode().split("\0") if p]


def scan(path: Path, secrets: list[str]) -> list[str]:
    if path.suffix.lower() in SKIP_SUFFIXES or path.name == ".env" or not path.is_file():
        return []
    try:
        text = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return []
    problems = [f"значение из .env ({secret[:4]}…)" for secret in secrets if secret in text]
    problems += [
        f"похоже на ключ API ({m.group()[:6]}…)"
        for p in KEY_PATTERNS
        for m in p.finditer(text)
        if looks_like_key(m.group())
    ]
    if path.suffix == ".ipynb" or "cassettes" in path.parts:
        problems += [f"путь с именем пользователя ({m.group()})" for m in HOME_PATH.finditer(text)]
    return problems


def main(argv: list[str]) -> int:
    files = [Path(a).resolve() for a in argv] if argv else tracked_files()
    secrets = env_secrets()
    found = 0
    for path in files:
        for problem in dict.fromkeys(scan(path, secrets)):
            found += 1
            print(f"✗ {path.relative_to(ROOT)}: {problem}")
    if found:
        print(f"\nНайдено проблем: {found}. Уберите секреты из файлов (ключи и адреса — только в .env).")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
