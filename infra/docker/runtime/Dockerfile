# Образ для запуска уроков курса: Python 3.12 + uv + системные утилиты.
# Окружения модулей собирает uv при старте контейнера из их uv.lock
# (make lab M=xx), поэтому образ один на весь курс.
FROM python:3.12-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PYTHON=/usr/local/bin/python3.12 \
    MLCOURSE_IN_DOCKER=1

RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential curl git \
 && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.12 /uv /uvx /usr/local/bin/

WORKDIR /course
