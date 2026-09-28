# Датасеты

Сами данные в git не хранятся. Уроки скачивают их через `mlcourse.fetch("<имя>")` с проверкой
SHA-256 в папку `data/` в корне репозитория (или в `MLCOURSE_DATA_DIR`). Реестр датасетов
с адресами и контрольными суммами — `shared/src/mlcourse/data.py`.

Курс использует только данные, лицензия которых допускает коммерческое использование
(CC0, CC BY, CC BY-SA, Apache 2.0, MIT). Тест `shared/tests/test_data.py` проверяет это для реестра.

| Датасет | Модули | Лицензия | Источник | Как загружается |
|---|---|---|---|---|
| digits (рукописные цифры 8×8) | M01 | CC BY 4.0 | UCI ML Repository | входит в scikit-learn |
| Adult (доход по данным переписи США) | M02 | CC BY 4.0 | [UCI ML Repository](https://archive.ics.uci.edu/dataset/2/adult) | `fetch("adult")` |
| RuReviews (отзывы о товарах, 3 класса) | M02, M04 | Apache 2.0 | [sismetanin/rureviews](https://github.com/sismetanin/rureviews) | `fetch("rureviews")` |
| Fashion-MNIST (одежда 28×28, 10 классов) | M03 | MIT | [zalandoresearch/fashion-mnist](https://github.com/zalandoresearch/fashion-mnist) | torchvision → `data_dir("fashion-mnist")` |
| EuroSAT (спутниковые снимки 64×64, 10 классов) | M03 | MIT | [phelber/EuroSAT](https://github.com/phelber/EuroSAT) | torchvision → `data_dir("eurosat")` |
| goodbooks-10k (оценки книг) | M05 | CC BY-SA 4.0 | [zygmuntz/goodbooks-10k](https://github.com/zygmuntz/goodbooks-10k) | `fetch("goodbooks-10k")` |

Датасеты, которые библиотеки скачивают сами, кладутся в `data_dir("<имя>")` и тоже перечислены здесь,
как только появляются в уроках.

## Предобученные модели

Лицензию каждой модели проверяем до того, как она появится в уроке.

| Модель | Модули | Лицензия |
|---|---|---|
| Qwen3 (`qwen3:8b` и др.) | все | Apache 2.0 |
| bge-m3 | M00, M01, M02, M11 | MIT |
| ResNet-18 `timm/resnet18.a1_in1k` | M03 | Apache 2.0 |
| CLIP ViT-B/32 `openai/clip-vit-base-patch32` | M03 | MIT (репозиторий openai/CLIP) |

