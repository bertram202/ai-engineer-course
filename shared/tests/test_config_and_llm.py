import pytest

from mlcourse import config, llm
from mlcourse.testing import fake_client, tool_call


@pytest.fixture(autouse=True)
def clean_env(monkeypatch, tmp_path):
    """Изолируемся от настоящего .env репозитория."""
    monkeypatch.chdir(tmp_path)
    for var in ("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL", "LLM_LOCAL_MODEL", "OLLAMA_BASE_URL", "EMBED_MODEL"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("LLM_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.delenv("MLCOURSE_IN_DOCKER", raising=False)


def test_main_defaults_to_local_ollama():
    s = config.load_settings()
    assert s.main.model == s.local.model == "qwen3:8b"
    assert s.main.base_url == "http://localhost:11434/v1"
    assert s.embed.model == "bge-m3"


def test_main_endpoint_from_env(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "https://llm.example.com/v1/")
    monkeypatch.setenv("LLM_API_KEY", "sk-test")
    monkeypatch.setenv("LLM_MODEL", "big-model")
    s = config.load_settings()
    assert (s.main.base_url, s.main.model) == ("https://llm.example.com/v1", "big-model")
    assert "sk-test" not in repr(s.main)


def test_localhost_is_rewritten_inside_docker(monkeypatch):
    monkeypatch.setenv("MLCOURSE_IN_DOCKER", "1")
    assert config.load_settings().local.base_url == "http://host.docker.internal:11434/v1"


def test_bad_cache_mode_is_rejected(monkeypatch):
    monkeypatch.setenv("LLM_CACHE", "sometimes")
    with pytest.raises(ValueError):
        config.load_settings()


def test_reasoning_text_supports_both_servers():
    assert llm.reasoning_text({"reasoning_content": "llama.cpp"}) == "llama.cpp"
    assert llm.reasoning_text({"reasoning": "ollama"}) == "ollama"
    assert llm.reasoning_text({}) == ""


def test_no_thinking_covers_ollama_and_llamacpp():
    params = llm.no_thinking()
    assert params["reasoning_effort"] == "none"
    assert params["chat_template_kwargs"] == {"enable_thinking": False}


def test_fake_client_returns_replies_in_order():
    client, requests = fake_client(["Привет!", tool_call("get_weather", city="Казань")])
    first = client.chat.completions.create(model="m", messages=[{"role": "user", "content": "hi"}])
    second = client.chat.completions.create(model="m", messages=[{"role": "user", "content": "погода?"}])
    assert first.choices[0].message.content == "Привет!"
    assert second.choices[0].message.tool_calls[0].function.name == "get_weather"
    assert requests[1]["messages"][0]["content"] == "погода?"


def test_safe_url_hides_remote_host_only():
    assert config.safe_url("http://localhost:11434/v1") == "http://localhost:11434/v1"
    masked = config.safe_url("https://secret.example.com/v1")
    assert "secret" not in masked and masked.endswith("/v1")


def test_describe_has_no_secrets(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "https://secret.example.com/v1")
    monkeypatch.setenv("LLM_API_KEY", "sk-top-secret")
    text = config.load_settings().describe()
    assert "secret" not in text


def test_repo_root(tmp_path):
    (tmp_path / "_quarto.yml").write_text("")
    nested = tmp_path / "modules" / "M00" / "lessons"
    nested.mkdir(parents=True)
    assert config.repo_root(nested) == tmp_path.resolve()
