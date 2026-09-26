"""Датасеты курса: загрузка с проверкой контрольных сумм.

Данные не хранятся в git. Уроки вызывают `fetch("rureviews")`: при первом вызове файлы
скачиваются в `data/<имя>/` в корне репозитория (или в `MLCOURSE_DATA_DIR`), проверяется
SHA-256, дальше берутся с диска. Для библиотек, которые качают данные сами
(torchvision, PyG, scikit-learn), есть `data_dir(имя)` — общая папка для их кэша.

Каждый датасет курса описан в `DATASETS` вместе с лицензией: курс использует только данные,
лицензия которых допускает коммерческое использование (см. `datasets/README.md`).
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from .config import repo_root


@dataclass(frozen=True)
class Remote:
    """Один файл датасета: откуда качать и какой у него SHA-256."""

    url: str
    sha256: str


@dataclass(frozen=True)
class Dataset:
    name: str
    title: str
    license: str
    source: str
    files: dict[str, Remote] = field(default_factory=dict)


_HF_RUREVIEWS = (
    "https://huggingface.co/datasets/ai-forever/ru-reviews-classification/resolve/"
    "0f67d914f396ce22917dc6463ec619799b3b08d2"
)
_GOODBOOKS = "https://raw.githubusercontent.com/zygmuntz/goodbooks-10k/6dd165b555a7b47b2dd36743a425776e641ff50c"

DATASETS: dict[str, Dataset] = {
    d.name: d
    for d in [
        Dataset(
            name="rureviews",
            title="RuReviews: отзывы о товарах (одежда), 3 класса тональности",
            license="Apache-2.0",
            source="https://github.com/sismetanin/rureviews",
            files={
                "train.jsonl": Remote(
                    f"{_HF_RUREVIEWS}/train.jsonl",
                    "0b97698a0c6871437d17e07c973018af9b8c9230ec9048d85cb875cc2c2470ea",
                ),
                "validation.jsonl": Remote(
                    f"{_HF_RUREVIEWS}/validation.jsonl",
                    "cb89b568a3b53e24976bd9e37476bff62fbbbd7991fc04284ecf342f38334031",
                ),
                "test.jsonl": Remote(
                    f"{_HF_RUREVIEWS}/test.jsonl",
                    "5ce9e33bcf14945012ca8405cf739844c1f13db0e4276d0481f8e25d9dec17b5",
                ),
            },
        ),
        Dataset(
            name="goodbooks-10k",
            title="goodbooks-10k: 6 млн оценок 10 тысяч книг",
            license="CC-BY-SA-4.0",
            source="https://github.com/zygmuntz/goodbooks-10k",
            files={
                "ratings.csv": Remote(
                    f"{_GOODBOOKS}/ratings.csv",
                    "1ee8a97172bd6d97147a30d9cfb3029ac240bd27ba2aa9365981c051c65113f8",
                ),
                "books.csv": Remote(
                    f"{_GOODBOOKS}/books.csv",
                    "4d21d0e0433128657c752e3a3d0405ac0ec11675d6c70c64cb0abf4ab1acb76a",
                ),
            },
        ),
    ]
}


def data_root() -> Path:
    """Корневая папка данных: `MLCOURSE_DATA_DIR` или `data/` в корне репозитория."""
    env = os.environ.get("MLCOURSE_DATA_DIR")
    return Path(env).expanduser() if env else repo_root() / "data"


def data_dir(name: str) -> Path:
    """Папка для данных с именем `name` (создаётся при необходимости)."""
    path = data_root() / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path, sha256: str, *, client: httpx.Client | None = None) -> Path:
    """Скачать `url` в `dest` и проверить SHA-256. Уже скачанный верный файл не качается заново.

    Файл пишется во временный `*.part` и переименовывается только после проверки суммы,
    поэтому оборванная загрузка не оставит «битый» файл под настоящим именем.
    """
    if dest.exists() and sha256_of(dest) == sha256:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    own = client is None
    client = client or httpx.Client(follow_redirects=True, timeout=httpx.Timeout(60, read=300))
    try:
        h = hashlib.sha256()
        with client.stream("GET", url) as r, part.open("wb") as f:
            r.raise_for_status()
            for chunk in r.iter_bytes(1 << 20):
                h.update(chunk)
                f.write(chunk)
        if h.hexdigest() != sha256:
            part.unlink(missing_ok=True)
            raise ValueError(f"{dest.name}: контрольная сумма не совпала — файл по ссылке изменился: {url}")
        part.replace(dest)
    finally:
        if own:
            client.close()
    return dest


def fetch(name: str, *, client: httpx.Client | None = None) -> Path:
    """Скачать (если ещё нет) все файлы датасета `name` и вернуть его папку."""
    if name not in DATASETS:
        raise KeyError(f"Неизвестный датасет {name!r}. Есть: {', '.join(sorted(DATASETS))}")
    ds = DATASETS[name]
    folder = data_dir(name)
    for filename, remote in ds.files.items():
        download(remote.url, folder / filename, remote.sha256, client=client)
    return folder
