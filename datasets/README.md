# Датасеты

Сами данные в git не хранятся. Уроки скачивают их через `mlcourse.fetch("<имя>")` с проверкой
SHA-256 в папку `data/` в корне репозитория (или в `MLCOURSE_DATA_DIR`). Реестр датасетов
с адресами и контрольными суммами — `shared/src/mlcourse/data.py`.

Курс использует только данные, лицензия которых допускает коммерческое использование
(CC0, CC BY, CC BY-SA, Apache 2.0, MIT). Тест `shared/tests/test_data.py` проверяет это для реестра.

| Датасет | Модули | Лицензия | Источник | Как загружается |
|---|---|---|---|---|
| digits (рукописные цифры 8×8) | M01 | CC BY 4.0 | UCI ML Repository | входит в scikit-learn |
| RuReviews (отзывы о товарах, 3 класса) | M02, M04 | Apache 2.0 | [sismetanin/rureviews](https://github.com/sismetanin/rureviews) | `fetch("rureviews")` |
| goodbooks-10k (оценки книг) | M05 | CC BY-SA 4.0 | [zygmuntz/goodbooks-10k](https://github.com/zygmuntz/goodbooks-10k) | `fetch("goodbooks-10k")` |

Датасеты, которые библиотеки скачивают сами, кладутся в `data_dir("<имя>")` и тоже перечислены здесь,
как только появляются в уроках.
