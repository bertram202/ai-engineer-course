# AGENTS.md — контекст для AI-агентов

Репозиторий курса «AI-инженер: от устройства LLM до мультиагентных систем».
Полный план — в `PLAN.md`, инструкции для студентов — в `README.md`.

## Структура

- `modules/Mxx-<slug>/` — модуль курса:
  - `theory/theory.qmd`, `theory/cheatsheet.qmd` — теория на Quarto (рендерится в PDF и HTML);
  - `lessons/NN-<slug>/lesson.ipynb` — уроки, `lessons/NN-<slug>/cassettes/` — записанные ответы LLM;
  - `exercises/*.py` — заготовки упражнений с `TODO`, `exercises/tests/` — автотесты;
  - `solutions/*.py` — эталонные решения;
  - `pyproject.toml` + `uv.lock` — отдельное uv-окружение модуля (Python 3.12).
- `shared/` — общая библиотека `mlcourse`: клиенты моделей (`get_client`, `get_chat_model`, `chat`, `embed`),
  кассеты ответов LLM, датасеты с проверкой SHA-256 (`fetch`, `data_dir`), выбор устройства PyTorch
  (`get_device`, `seed_everything`), фейковый OpenAI-сервер для тестов (`mlcourse.testing`).
- `infra/` — Docker-образ `runtime` и docker compose (JupyterLab + сервисы).
- `quarto/` — стиль PDF (LaTeX-преамбула) и сайта (SCSS).
- `templates/module/` — заготовка нового модуля (`make new-module`).

## Команды

- `make help` — все команды.
- `make test M=00` — тесты упражнений (на коде студента); `make test-solutions M=00` — на эталоне.
- `make check M=00` — выполнить все ноутбуки модуля.
- `make pdf M=00` — PDF теории; `make site` — превью сайта.
- `cd shared && uv run pytest` — тесты библиотеки.

## Правила

- **Секреты.** Никогда не коммить `.env`, ключи API и адреса удалённых серверов моделей.
  В коде и выводах ноутбуков — только `mlcourse.describe_settings()` (адрес маскируется).
  Абсолютные пути с именем пользователя в выводы ноутбуков не попадают.
- **Модели.** Код уроков обращается к моделям только через `mlcourse` (роли `main`, `local`, `embed`),
  без хардкода адресов и имён моделей. Режим рассуждений в агентных циклах выключен: `extra_body=no_thinking()`.
- **Кассеты.** Ноутбуки записываются с `LLM_CACHE=record` (`make record M=xx`); CI выполняет их с `LLM_CACHE=replay`.
- **Язык.** Тексты — на русском, английские термины в скобках при первом упоминании;
  идентификаторы в коде — на английском, комментарии — на русском.
- **Стиль кода.** Python 3.12, type hints, `ruff`. Код с нуля — понятный, а не хитрый.
- **Упражнения.** Пиши решение в `solutions/`, заготовку генерирует `python3 scripts/make_exercises.py modules/Mxx`
  (тела функций → `NotImplementedError`, строка `# подсказка: …` → текст подсказки). Заготовки падают на тестах,
  решения проходят. Не раскрывай решения студенту.
- **Данные.** Только датасеты с лицензией, разрешающей коммерческое использование; новый датасет — в реестр
  `shared/src/mlcourse/data.py` и в `datasets/README.md`.
- **Рисунки теории** генерирует `theory/figures/make_figures.py` модуля (стиль `quarto/figures.mplstyle`).
- **Формат модуля** описан в `PLAN.md`, раздел 4. Эталон — модуль `M00-setup`.
