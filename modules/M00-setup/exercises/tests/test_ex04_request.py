from ex04_request import build_chat_request


def test_minimal_request():
    req = build_chat_request("qwen3:8b", "Привет", thinking=True)
    assert req == {"model": "qwen3:8b", "messages": [{"role": "user", "content": "Привет"}]}


def test_system_history_user_order():
    history = [{"role": "user", "content": "1"}, {"role": "assistant", "content": "2"}]
    req = build_chat_request("m", "3", system="Будь краток", history=history, thinking=True)
    assert [m["role"] for m in req["messages"]] == ["system", "user", "assistant", "user"]
    assert req["messages"][-1]["content"] == "3"


def test_history_is_not_mutated():
    history = [{"role": "user", "content": "1"}]
    build_chat_request("m", "2", history=history)
    assert history == [{"role": "user", "content": "1"}]


def test_thinking_off_by_default_for_both_servers():
    req = build_chat_request("m", "hi")
    assert req["reasoning_effort"] == "none"
    assert req["chat_template_kwargs"] == {"enable_thinking": False}


def test_thinking_on_adds_nothing():
    req = build_chat_request("m", "hi", thinking=True)
    assert "reasoning_effort" not in req and "chat_template_kwargs" not in req


def test_json_schema():
    schema = {"type": "object", "properties": {"x": {"type": "integer"}}}
    req = build_chat_request("m", "hi", json_schema=schema)
    assert req["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "response", "schema": schema},
    }


def test_max_tokens_only_when_set():
    assert "max_tokens" not in build_chat_request("m", "hi")
    assert build_chat_request("m", "hi", max_tokens=50)["max_tokens"] == 50
