# M00. Старт: окружение, модели и инструменты

🎯 **Цель модуля:** за вечер получить рабочее место и проверенные модели, а заодно разобраться в понятиях,
которые пройдут через весь курс: OpenAI-совместимый API, токены и контекст, скорость генерации,
память модели, кассеты с ответами LLM.

⏱ ~4 часа · **Пререквизиты:** уверенный Python, терминал, git.

## Материалы

| | Материал | О чём |
|---|---|---|
| 📖 | [Теория](theory/theory.qmd) (PDF — `make pdf M=00`) | AI-инженер, окружение, контейнеры, API моделей, скорость и память, кассеты |
| 📋 | [Шпаргалка](theory/cheatsheet.qmd) | команды, `.env`, формулы |
| 💻 | [Урок 1. Окружение курса](lessons/01-environment/lesson.ipynb) | uv, lock-файлы, библиотека `mlcourse`, `.env` |
| 💻 | [Урок 2. Docker и сервисы](lessons/02-docker/lesson.ipynb) | как устроен `make lab`, `host.docker.internal`, сервисы |
| 💻 | [Урок 3. Первый разговор с моделью](lessons/03-models/lesson.ipynb) | сырой HTTP, SDK, стриминг, рассуждения, tools, эмбеддинги, память и скорость |
| 💻 | [Урок 4. OpenCode](lessons/04-opencode/lesson.ipynb) | конфигурация, агенты и команды курса |
| ✍️ | [Упражнения](exercises/) | память модели, скорость, разбор SSE, сборка запроса |

## Настройка (один раз)

```bash
make setup          # .env из шаблона, Python 3.12, библиотека курса, pre-commit
make ollama-pull    # qwen3:8b (~5 ГБ) и bge-m3 (~1 ГБ)
make ollama         # сервер Ollama с контекстом 16K — держите терминал открытым
make llm-check      # проверка моделей
```

Если у вас есть свой OpenAI-совместимый сервер (vLLM, llama.cpp) или облачный провайдер, укажите в `.env`
`LLM_BASE_URL`, `LLM_API_KEY` и `LLM_MODEL` — это станет основной моделью (роль `main`).

## Запуск уроков

```bash
make native M=00    # нативно (рекомендуется на Mac)
make lab M=00       # в Docker → http://localhost:8888
make test M=00      # проверка упражнений
make opencode       # OpenCode с агентом-репетитором
```

## Если что-то не работает

| Симптом | Причина и решение |
|---|---|
| `llm-check`: ❌ сервер и модель, ошибка соединения | Ollama не запущена: `make ollama` |
| `llm-check`: ❌ контекст Ollama 4096 < 16384 | Ollama запущена без `OLLAMA_CONTEXT_LENGTH`: остановите её и запустите `make ollama` |
| `модели 'qwen3:8b' нет на сервере` | `make ollama-pull` |
| Ноутбук: `Kernel died before replying to kernel_info` | запускайте через `make native`: он выбирает ядро из окружения модуля, а не из `~/Library/Jupyter/kernels` |
| OpenCode: `exceeds the available context size` | модели нужно окно 32K+: `OLLAMA_CONTEXT_LENGTH=32768` или флаг `-c` вашего сервера |
| В Docker нет связи с Ollama | на Linux Ollama должна слушать `0.0.0.0`: `OLLAMA_HOST=0.0.0.0 make ollama` |

## ✅ Я умею

- [ ] запускать уроки нативно и в Docker;
- [ ] объяснить, чем `pyproject.toml` отличается от `uv.lock`;
- [ ] отправить запрос к модели «сырым» HTTP и через openai SDK, получить потоковый ответ;
- [ ] оценить память модели и предел скорости генерации;
- [ ] объяснить, зачем нужны кассеты и чем отличаются режимы `LLM_CACHE`;
- [ ] работать с OpenCode и агентом-репетитором.
