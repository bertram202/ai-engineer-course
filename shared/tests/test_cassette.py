import json

import httpx
import httpx2
import pytest
from openai import AsyncOpenAI, OpenAI

from mlcourse.cassette import (
    AsyncCassetteTransport,
    AsyncSDKCassetteTransport,
    CassetteMissError,
    CassetteStore,
    CassetteTransport,
    SDKCassetteTransport,
)
from mlcourse.testing import chat_completion

# один и тот же код кассет должен работать и с httpx (сырые запросы), и с httpx2 (openai SDK 3.x)
FLAVORS = {
    "httpx": (httpx, CassetteTransport, AsyncCassetteTransport),
    "httpx2": (httpx2, SDKCassetteTransport, AsyncSDKCassetteTransport),
}


@pytest.fixture(params=list(FLAVORS))
def flavor(request):
    return FLAVORS[request.param]

SECRET_HOST = "secret-host.example"
SECRET_KEY = "sk-very-secret"


class CountingServer:
    """Фейковый сервер: считает запросы, умеет отвечать обычным JSON и SSE-потоком."""

    lib = httpx

    def __init__(self, reply: str = "ответ"):
        self.calls = 0
        self.reply = reply

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.calls += 1
        body = json.loads(request.content)
        if body.get("stream"):
            chunks = [
                {"choices": [{"index": 0, "delta": {"content": part}}]} for part in (self.reply[:3], self.reply[3:])
            ]
            sse = "".join(f"data: {json.dumps(c, ensure_ascii=False)}\n\n" for c in chunks) + "data: [DONE]\n\n"
            return self.lib.Response(200, headers={"content-type": "text/event-stream"}, content=sse.encode())
        return self.lib.Response(200, json=chat_completion(f"{self.reply} #{self.calls}", model=body["model"]))


_current_flavor = FLAVORS["httpx2"]


@pytest.fixture(autouse=True)
def _use_flavor(flavor):
    global _current_flavor
    _current_flavor = flavor


def make_client(server, tmp_path, mode, role="main", model="model-a"):
    lib, transport_cls, _ = _current_flavor
    server.lib = lib
    store = CassetteStore(tmp_path / "cassettes", tmp_path / "cache", mode)
    transport = transport_cls(lib.MockTransport(server), store, role=role, base_path="/v1")
    client = OpenAI(
        base_url=f"https://{SECRET_HOST}/v1",
        api_key=SECRET_KEY,
        max_retries=0,
        http_client=lib.Client(transport=transport),
    )

    def ask(text="привет", **kw):
        return client.chat.completions.create(model=model, messages=[{"role": "user", "content": text}], **kw)

    return ask


def test_on_mode_caches_in_personal_cache(tmp_path):
    server = CountingServer()
    ask = make_client(server, tmp_path, "on")
    first = ask().choices[0].message.content
    second = ask().choices[0].message.content
    assert server.calls == 1
    assert first == second == "ответ #1"
    assert len(list((tmp_path / "cache").glob("*.json"))) == 1
    assert not (tmp_path / "cassettes").exists()


def test_different_requests_are_different_records(tmp_path):
    server = CountingServer()
    ask = make_client(server, tmp_path, "on")
    ask("раз")
    ask("два")
    assert server.calls == 2


def test_record_then_replay_without_network(tmp_path):
    server = CountingServer()
    make_client(server, tmp_path, "record")()
    offline = CountingServer()
    reply = make_client(offline, tmp_path, "replay")().choices[0].message.content
    assert offline.calls == 0
    assert reply == "ответ #1"


def test_record_copies_hits_from_personal_cache(tmp_path):
    server = CountingServer()
    make_client(server, tmp_path, "on")()
    make_client(server, tmp_path, "record")()
    assert server.calls == 1
    assert len(list((tmp_path / "cassettes").glob("*.json"))) == 1


def test_refresh_always_calls_model(tmp_path):
    server = CountingServer()
    make_client(server, tmp_path, "refresh")()
    make_client(server, tmp_path, "refresh")()
    assert server.calls == 2


def test_replay_falls_back_to_same_role_on_other_model(tmp_path):
    make_client(CountingServer(), tmp_path, "record", model="model-a")()
    reply = make_client(CountingServer(), tmp_path, "replay", model="model-b")()
    assert reply.choices[0].message.content == "ответ #1"


def test_replay_does_not_mix_roles(tmp_path):
    make_client(CountingServer(), tmp_path, "record", role="main", model="model-a")()
    with pytest.raises(Exception) as err:
        make_client(CountingServer(), tmp_path, "replay", role="local", model="model-b")()
    assert isinstance(err.value.__cause__, CassetteMissError) or "Нет записи" in str(err.value)


def test_off_mode_never_touches_disk(tmp_path):
    server = CountingServer()
    ask = make_client(server, tmp_path, "off")
    ask()
    ask()
    assert server.calls == 2
    assert not (tmp_path / "cache").exists() and not (tmp_path / "cassettes").exists()


def test_streaming_is_recorded_and_replayed(tmp_path):
    ask = make_client(CountingServer("потоковый"), tmp_path, "record")
    live = "".join(c.choices[0].delta.content or "" for c in ask(stream=True) if c.choices)
    offline = make_client(CountingServer(), tmp_path, "replay")
    replayed = "".join(c.choices[0].delta.content or "" for c in offline(stream=True) if c.choices)
    assert live == replayed == "потоковый"


def test_records_contain_no_host_and_no_key(tmp_path):
    make_client(CountingServer(), tmp_path, "record")()
    text = next((tmp_path / "cassettes").glob("*.json")).read_text(encoding="utf-8")
    assert SECRET_HOST not in text and SECRET_KEY not in text
    assert json.loads(text)["request"]["path"] == "chat/completions"


async def test_async_client_uses_cassettes(tmp_path, flavor):
    lib, _, async_cls = flavor
    server = CountingServer()
    server.lib = lib
    store = CassetteStore(tmp_path / "cassettes", tmp_path / "cache", "on")
    transport = async_cls(lib.MockTransport(server), store, role="main", base_path="/v1")
    client = AsyncOpenAI(base_url="https://h/v1", api_key="k", http_client=lib.AsyncClient(transport=transport))
    for _ in range(2):
        await client.chat.completions.create(model="m", messages=[{"role": "user", "content": "hi"}])
    assert server.calls == 1
