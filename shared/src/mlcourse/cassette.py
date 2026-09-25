"""Кассеты: запись и воспроизведение ответов LLM на уровне HTTP.

Зачем:
  * повторный запуск ноутбука мгновенный и не тратит время медленной модели;
  * ноутбуки выполняются в публичном CI, где нет доступа к моделям;
  * работает с любым клиентом поверх httpx: openai SDK, LangChain, LlamaIndex…

Где хранится:
  * кассеты урока — `./cassettes/` рядом с ноутбуком (коммитятся в git);
  * личный кэш — `~/.cache/mlcourse/llm/` (не коммитится).

Режимы (переменная LLM_CACHE):
  off     — без кэша, всегда реальный запрос;
  on      — читать кассеты и личный кэш, новые ответы писать в личный кэш (по умолчанию);
  record  — как on, но всё использованное пишется в кассеты урока;
  refresh — всегда реальный запрос, ответ пишется в кассеты урока;
  replay  — только кассеты, без сети (CI). Если точной записи нет, ищется запись
            для той же роли модели (main/local) — так кассеты, записанные на одной
            модели, воспроизводятся при любой другой настройке `.env`.

В кассету попадают только путь запроса (без хоста), тело запроса и тело ответа.
URL сервера, заголовки и ключи API не сохраняются никогда.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import AsyncIterator, Callable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import Any

import httpx


class CassetteMissError(RuntimeError):
    """В режиме replay нужной записи нет в кассетах."""


def _canonical_body(raw: bytes) -> Any:
    if not raw:
        return None
    try:
        return json.loads(raw)
    except ValueError:
        return raw.decode("utf-8", errors="replace")


def _digest(obj: Any) -> str:
    text = json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class RequestKey:
    exact: str  # запрос целиком, включая имя модели
    role: str  # тот же запрос, но модель заменена ролью (для replay на другой модели)
    method: str
    path: str
    body: Any


def make_key(request: Any, *, role: str, base_path: str) -> RequestKey:
    path = request.url.path
    if base_path and path.startswith(base_path):
        path = path[len(base_path) :]
    path = path.lstrip("/")
    body = _canonical_body(request.content)
    exact = _digest({"method": request.method, "path": path, "body": body})
    role_body = dict(body, model=f"<{role}>") if isinstance(body, dict) and "model" in body else body
    by_role = _digest({"method": request.method, "path": path, "body": role_body})
    return RequestKey(exact=exact[:16], role=by_role[:16], method=request.method, path=path, body=body)


@dataclass
class Recording:
    status: int
    content_type: str
    body: bytes

    def to_response(self, request: Any, lib: ModuleType = httpx) -> Any:
        return lib.Response(
            self.status,
            headers={"content-type": self.content_type, "x-mlcourse-cassette": "hit"},
            content=self.body,
            request=request,
        )


class CassetteStore:
    """Хранилище записей: кассеты урока + личный кэш пользователя."""

    def __init__(self, cassette_dir: Path, cache_dir: Path, mode: str):
        self.cassette_dir = Path(cassette_dir)
        self.cache_dir = Path(cache_dir)
        self.mode = mode

    # --- политика режимов -------------------------------------------------
    @property
    def reads(self) -> bool:
        return self.mode in ("on", "record", "replay")

    @property
    def may_call(self) -> bool:
        return self.mode != "replay"

    @property
    def write_dir(self) -> Path | None:
        if self.mode in ("record", "refresh"):
            return self.cassette_dir
        if self.mode == "on":
            return self.cache_dir
        return None

    # --- чтение -----------------------------------------------------------
    def lookup(self, key: RequestKey) -> Recording | None:
        if not self.reads:
            return None
        dirs = [self.cassette_dir] if self.mode == "replay" else [self.cassette_dir, self.cache_dir]
        for d in dirs:
            found = sorted(d.glob(f"*-{key.exact}.json")) if d.is_dir() else []
            if found:
                return self._load(found[-1], key)
        if self.mode == "replay" and self.cassette_dir.is_dir():
            found = sorted(self.cassette_dir.glob(f"{key.role}-*.json"))
            if found:
                return self._load(found[-1], key)
        return None

    def _load(self, file: Path, key: RequestKey) -> Recording:
        data = json.loads(file.read_text(encoding="utf-8"))
        resp = data["response"]
        rec = Recording(resp["status"], resp["content_type"], resp["body"].encode("utf-8"))
        # record: всё, что прочитали из личного кэша, переносим в кассеты урока
        if self.mode == "record" and file.parent != self.cassette_dir:
            self._write(self.cassette_dir, key, rec, data.get("recorded_at"))
        return rec

    # --- запись -----------------------------------------------------------
    def save(self, key: RequestKey, rec: Recording) -> None:
        if (target := self.write_dir) is not None and rec.status < 400:
            self._write(target, key, rec)

    def _write(self, target: Path, key: RequestKey, rec: Recording, when: str | None = None) -> None:
        target.mkdir(parents=True, exist_ok=True)
        payload = {
            "recorded_at": when or datetime.now(UTC).isoformat(timespec="seconds"),
            "request": {"method": key.method, "path": key.path, "body": key.body},
            "response": {
                "status": rec.status,
                "content_type": rec.content_type,
                "body": rec.body.decode("utf-8", errors="replace"),
            },
        }
        file = target / f"{key.role}-{key.exact}.json"
        tmp = file.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
        os.replace(tmp, file)


# --- транспорты ------------------------------------------------------------
#
# openai SDK 3.x работает поверх httpx2 (форк httpx), а многие другие библиотеки —
# поверх классического httpx. API у них одинаковый, но классы разные, поэтому
# транспорты собираются фабрикой для каждой библиотеки.


class _Recorder:
    """Копит байты ответа и один раз сохраняет их, когда ответ получен целиком.

    openai SDK закрывает SSE-поток сразу после `data: [DONE]`, не дочитывая его
    до конца, поэтому завершённый поток распознаём и при закрытии.
    """

    def __init__(self, on_complete: Callable[[bytes], None]):
        self._on_complete = on_complete
        self.buf = bytearray()
        self.saved = False

    def finish(self, *, exhausted: bool) -> None:
        if self.saved:
            return
        if exhausted or bytes(self.buf).rstrip().endswith(b"data: [DONE]"):
            self.saved = True
            self._on_complete(bytes(self.buf))


class _Base:
    lib: ModuleType

    def __init__(self, store: CassetteStore, *, role: str, base_path: str):
        self.store, self.role, self.base_path = store, role, base_path.rstrip("/")

    def _prepare(self, request: Any) -> tuple[RequestKey, Recording | None]:
        # без сжатия: записываем и отдаём «чистые» байты
        request.headers["accept-encoding"] = "identity"
        key = make_key(request, role=self.role, base_path=self.base_path)
        hit = self.store.lookup(key)
        if hit is None and not self.store.may_call:
            raise CassetteMissError(
                f"Нет записи в кассетах для {key.method} /{key.path} (роль {self.role}). "
                f"Папка кассет: {self.store.cassette_dir}. Запустите ноутбук с LLM_CACHE=on или record."
            )
        return key, hit

    def _saver(self, key: RequestKey, response: Any) -> Callable[[bytes], None]:
        ctype = response.headers.get("content-type", "application/json")
        return lambda body: self.store.save(key, Recording(response.status_code, ctype, body))

    def _wrap(self, request: Any, response: Any, stream: Any) -> Any:
        drop = {"content-encoding", "content-length", "transfer-encoding"}
        return self.lib.Response(
            response.status_code,
            headers={k: v for k, v in response.headers.items() if k.lower() not in drop},
            stream=stream,
            request=request,
            extensions=response.extensions,
        )


def _build_transports(lib: ModuleType) -> tuple[type, type]:
    class RecordingStream(lib.SyncByteStream):
        """Отдаёт поток клиенту без задержки (стриминг токенов не ломается) и копит его для записи."""

        def __init__(self, inner: Any, on_complete: Callable[[bytes], None]):
            self._inner, self._rec = inner, _Recorder(on_complete)

        def __iter__(self) -> Iterator[bytes]:
            for chunk in self._inner.stream:
                self._rec.buf.extend(chunk)
                yield chunk
            self._rec.finish(exhausted=True)

        def close(self) -> None:
            self._rec.finish(exhausted=False)
            self._inner.close()

    class AsyncRecordingStream(lib.AsyncByteStream):
        def __init__(self, inner: Any, on_complete: Callable[[bytes], None]):
            self._inner, self._rec = inner, _Recorder(on_complete)

        async def __aiter__(self) -> AsyncIterator[bytes]:
            async for chunk in self._inner.stream:
                self._rec.buf.extend(chunk)
                yield chunk
            self._rec.finish(exhausted=True)

        async def aclose(self) -> None:
            self._rec.finish(exhausted=False)
            await self._inner.aclose()

    class CassetteTransport(_Base, lib.BaseTransport):
        """Синхронный транспорт с кассетами поверх любого транспорта `inner`."""

        def __init__(self, inner: Any, store: CassetteStore, *, role: str, base_path: str = ""):
            super().__init__(store, role=role, base_path=base_path)
            self.inner = inner

        def handle_request(self, request: Any) -> Any:
            request.read()
            key, hit = self._prepare(request)
            if hit is not None:
                return hit.to_response(request, lib)
            response = self.inner.handle_request(request)
            if self.store.write_dir is None or response.status_code >= 400:
                return response
            return self._wrap(request, response, RecordingStream(response, self._saver(key, response)))

        def close(self) -> None:
            self.inner.close()

    class AsyncCassetteTransport(_Base, lib.AsyncBaseTransport):
        """Асинхронный транспорт с кассетами поверх любого транспорта `inner`."""

        def __init__(self, inner: Any, store: CassetteStore, *, role: str, base_path: str = ""):
            super().__init__(store, role=role, base_path=base_path)
            self.inner = inner

        async def handle_async_request(self, request: Any) -> Any:
            await request.aread()
            key, hit = self._prepare(request)
            if hit is not None:
                return hit.to_response(request, lib)
            response = await self.inner.handle_async_request(request)
            if self.store.write_dir is None or response.status_code >= 400:
                return response
            return self._wrap(request, response, AsyncRecordingStream(response, self._saver(key, response)))

        async def aclose(self) -> None:
            await self.inner.aclose()

    for cls in (CassetteTransport, AsyncCassetteTransport):
        cls.lib = lib
        cls.__qualname__ = cls.__name__
    return CassetteTransport, AsyncCassetteTransport


# классический httpx: «сырые» запросы в уроках и библиотеки поверх httpx
CassetteTransport, AsyncCassetteTransport = _build_transports(httpx)

# httpx2: openai SDK 3.x и всё, что поверх него (LangChain и др.)
try:
    import httpx2
except ImportError:  # старый openai SDK без httpx2
    httpx2 = None
    SDKCassetteTransport, AsyncSDKCassetteTransport = CassetteTransport, AsyncCassetteTransport
else:
    SDKCassetteTransport, AsyncSDKCassetteTransport = _build_transports(httpx2)
