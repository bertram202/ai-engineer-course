import hashlib

import httpx
import pytest

from mlcourse import data
from mlcourse.data import Dataset, Remote, download, fetch

PAYLOAD = b"user_id,book_id,rating\n1,258,5\n"
SHA = hashlib.sha256(PAYLOAD).hexdigest()


def mock_client(calls: list[str], payload: bytes = PAYLOAD) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        return httpx.Response(200, content=payload)

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_download_checks_hash_and_caches(tmp_path):
    calls: list[str] = []
    dest = tmp_path / "ratings.csv"
    assert download("https://example.org/r.csv", dest, SHA, client=mock_client(calls)) == dest
    assert dest.read_bytes() == PAYLOAD
    download("https://example.org/r.csv", dest, SHA, client=mock_client(calls))
    assert len(calls) == 1  # второй раз файл берётся с диска


def test_download_rejects_wrong_hash(tmp_path):
    dest = tmp_path / "ratings.csv"
    with pytest.raises(ValueError, match="контрольная сумма"):
        download("https://example.org/r.csv", dest, "0" * 64, client=mock_client([]))
    assert not dest.exists()
    assert not list(tmp_path.iterdir())  # и временный .part удалён


def test_download_replaces_corrupted_file(tmp_path):
    dest = tmp_path / "ratings.csv"
    dest.write_text("битый файл")
    calls: list[str] = []
    download("https://example.org/r.csv", dest, SHA, client=mock_client(calls))
    assert calls and dest.read_bytes() == PAYLOAD


def test_fetch_uses_registry_and_data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("MLCOURSE_DATA_DIR", str(tmp_path))
    monkeypatch.setitem(
        data.DATASETS,
        "toy",
        Dataset("toy", "игрушка", "CC0-1.0", "тест", {"r.csv": Remote("https://example.org/r.csv", SHA)}),
    )
    folder = fetch("toy", client=mock_client([]))
    assert folder == tmp_path / "toy"
    assert (folder / "r.csv").read_bytes() == PAYLOAD


def test_fetch_unknown_dataset():
    with pytest.raises(KeyError, match="Неизвестный датасет"):
        fetch("нет-такого")


def test_registry_is_license_clean():
    allowed = {"CC0-1.0", "CC-BY-4.0", "CC-BY-SA-4.0", "Apache-2.0", "MIT"}
    for ds in data.DATASETS.values():
        assert ds.license in allowed, ds.name
        assert all(len(r.sha256) == 64 for r in ds.files.values())


def test_seed_everything_is_reproducible():
    import numpy as np

    from mlcourse import seed_everything

    seed_everything(1)
    a = np.random.rand(3)
    seed_everything(1)
    assert (np.random.rand(3) == a).all()
