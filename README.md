# AI-инженер: от устройства LLM до мультиагентных систем

Практический курс для Python-разработчиков, которые хотят строить продукты на больших языковых моделях:
от устройства трансформера до RAG-систем, мультиагентных приложений на **LangChain** и **LangGraph**,
**MCP**-серверов и coding-агентов вроде **OpenCode**.

- **28 модулей** и 4 капстоун-проекта, ~410 часов;
- теория в PDF и на [сайте курса](https://bertram202.github.io/ai-engineer-course) — с выводами формул, а не «легко видеть, что»;
- уроки в Jupyter: сначала пишем с нуля, потом берём фреймворк и сравниваем;
- упражнения с автотестами, мини-проект в каждом модуле;
- **без платных API**: всё работает на локальной модели в Ollama или на любом OpenAI-совместимом сервере.

Подробная программа — в [PLAN.md](PLAN.md).

## Быстрый старт

Нужны: macOS или Linux, 16 ГБ RAM, [uv](https://docs.astral.sh/uv/), [Ollama](https://ollama.com),
[Docker](https://docs.docker.com/get-docker/) (по желанию) и [Quarto](https://quarto.org) (для сборки PDF, по желанию).

```bash
git clone https://github.com/bertram202/ai-engineer-course.git
cd ai-engineer-course
make setup          # .env, Python 3.12, библиотека курса, pre-commit
make ollama-pull    # модели qwen3:8b и bge-m3
make ollama         # сервер Ollama с контекстом 16K — в отдельном терминале
make llm-check      # проверка: чат, скорость, tool calling, JSON Schema, эмбеддинги
make native M=00    # первый модуль в JupyterLab
```

Дальше — [модуль M00](modules/M00-setup/README.md): он объясняет, как всё устроено.

## Как устроен курс

```text
modules/Mxx-<тема>/
├── theory/            теория (PDF и страница сайта) + шпаргалка
├── lessons/NN-*/      уроки-ноутбуки и записанные ответы моделей (cassettes/)
├── exercises/         упражнения с TODO и автотесты
├── solutions/         эталонные решения
└── project/           мини-проект модуля
```

| Команда | Что делает |
|---|---|
| `make native M=06` | JupyterLab модуля нативно (нужно для PyTorch на GPU Apple) |
| `make lab M=15` | JupyterLab модуля в Docker → http://localhost:8888 |
| `make test M=06` | проверить свои упражнения |
| `make pdf M=06` | собрать PDF теории |
| `make services S="qdrant postgres"` | поднять сервисы для модуля |
| `make opencode` | AI-напарник OpenCode с моделями курса |
| `make help` | все команды |

### Модели

Настройки — в `.env` (шаблон — [.env.example](.env.example)). По умолчанию всё идёт в локальную Ollama.
Если у вас есть свой сервер или облачный провайдер с OpenAI-совместимым API, укажите его в
`LLM_BASE_URL`, `LLM_API_KEY` и `LLM_MODEL` — код уроков менять не нужно.

Ответы моделей в уроках записаны в «кассеты», поэтому ноутбуки открываются с результатами и
перезапускаются мгновенно. Подробно — в теории M00.

## Программа

| Часть | Модули |
|---|---|
| 0. Старт | M00 окружение, модели и инструменты |
| I. Фундамент-экспресс | M01 математика · M02 классическое ML · M03 нейросети и PyTorch · M04 NLP · M05 обзор остального ML |
| II. Устройство LLM | M06 трансформер изнутри · M07 как обучают LLM · M08 инференс |
| III. Работа с LLM | M09 промпты и контекст · M10 structured output и tool calling · M11 эмбеддинги · M12–M13 RAG |
| IV. Агенты | M14 агенты с нуля · M15 LangChain · M16–M17 LangGraph · M18 MCP · M19 фреймворки · M20 продвинутые агенты · M21 OpenCode |
| V. Надёжность | M22 оценка качества · M23 observability и безопасность |
| VI. Продакшн | M24 LLM-сервисы · M25 fine-tuning · M26 мультимодальность · M27 MLOps |
| VII. Капстоуны | RAG-ассистент · мультиагентная система · свой coding-агент · свой продукт |

## Лицензия

Тексты курса — [CC BY-NC-SA 4.0](LICENSE), код — [MIT](LICENSE-CODE).
