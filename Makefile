# Единая точка входа курса «AI-инженер». Список команд: make help
SHELL := /bin/bash
.DEFAULT_GOAL := help

M ?= 00
S ?= qdrant
MOD := $(firstword $(wildcard modules/M$(M)-*))
COMPOSE := docker compose -f infra/compose/compose.yml

define require_module
	@test -n "$(MOD)" || { echo "Модуль M$(M) не найден в modules/"; exit 1; }
endef

.PHONY: help setup ollama ollama-pull llm-check native lab test test-solutions check record \
        pdf site site-build services services-down opencode secrets lint new-module clean

help: ## Показать список команд
	@awk 'BEGIN{FS=":.*## "} /^[a-zA-Z_-]+:.*## /{printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

setup: ## Первичная настройка: .env, Python 3.12, библиотека курса, pre-commit
	@test -f .env || { cp .env.example .env && echo "Создан .env из .env.example"; }
	uv python install 3.12
	cd shared && uv sync
	uvx pre-commit install
	@echo "Готово. Дальше: make ollama-pull, make ollama (в отдельном терминале), make llm-check"

ollama: ## Запустить Ollama с контекстом 16K (держите терминал открытым)
	OLLAMA_CONTEXT_LENGTH=16384 ollama serve

ollama-pull: ## Скачать модели курса в Ollama
	ollama pull qwen3:8b
	ollama pull bge-m3

llm-check: ## Проверить модели из .env: чат, скорость, tools, JSON Schema, эмбеддинги
	cd shared && uv run python -m mlcourse.check

native: ## JupyterLab модуля нативно на Mac: make native M=00
	$(require_module)
	scripts/jupyter.sh $(MOD) lab

lab: ## JupyterLab модуля в Docker: make lab M=00 → http://localhost:8888
	$(require_module)
	MODULE=$(MOD) MODULE_NAME=$(notdir $(MOD)) $(COMPOSE) run --rm --build --service-ports lab

test: ## Проверить свои упражнения: make test M=00
	$(require_module)
	cd $(MOD) && uv run pytest -q

test-solutions: ## Проверить эталонные решения
	$(require_module)
	cd $(MOD) && MLCOURSE_TARGET=solutions uv run pytest -q

check: ## Выполнить все ноутбуки модуля (ответы LLM — из кассет и кэша)
	$(require_module)
	scripts/run_notebooks.sh $(MOD)

record: ## Записать уроки на реальных моделях: выходы ноутбуков + кассеты
	$(require_module)
	LLM_CACHE=record scripts/run_notebooks.sh $(MOD) --inplace

pdf: ## PDF теории и шпаргалки модуля: make pdf M=00 → _site/…
	$(require_module)
	quarto render $(MOD)/theory/theory.qmd --to pdf
	quarto render $(MOD)/theory/cheatsheet.qmd --to pdf

site: ## Превью сайта курса с автообновлением
	quarto preview

site-build: ## Собрать сайт и все PDF в _site/
	quarto render

services: ## Поднять сервисы: make services S="qdrant postgres"
	$(COMPOSE) $(foreach s,$(S),--profile $(s)) up -d $(S)

services-down: ## Остановить все сервисы
	$(COMPOSE) --profile '*' down

opencode: ## OpenCode с моделями из .env
	set -a && source ./.env && set +a && opencode

secrets: ## Проверить репозиторий на утёкшие секреты
	python3 scripts/check_secrets.py

lint: ## Проверить стиль кода (ruff)
	uvx ruff check shared modules scripts

new-module: ## Новый модуль из шаблона: make new-module M=01 NAME=math TITLE="Математика для LLM"
	@test -n "$(NAME)" || { echo "Укажите NAME=<slug>"; exit 1; }
	@test ! -e modules/M$(M)-$(NAME) || { echo "modules/M$(M)-$(NAME) уже существует"; exit 1; }
	cp -R templates/module modules/M$(M)-$(NAME)
	find modules/M$(M)-$(NAME) -type f -exec perl -CSD -Mutf8 -pi -e \
		's/__ID__/$(M)/g; s/__SLUG__/$(NAME)/g; s/__TITLE__/$(or $(TITLE),$(NAME))/g' {} +
	cd modules/M$(M)-$(NAME) && uv lock -q
	@echo "Создан modules/M$(M)-$(NAME). Добавьте его в sidebar в _quarto.yml."

clean: ## Удалить артефакты сборки сайта
	rm -rf _site .quarto
