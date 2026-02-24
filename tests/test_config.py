from app.core.config import AppSettings


def test_default_values():
    s = AppSettings()
    assert s.app_name == "LLM-Proxy"
    assert s.root_path == "/ai"
    assert s.debug is False
    assert s.port == 5000
    assert s.proxy_enabled is False
    assert s.prompt_hub_enabled is False
    assert s.chain_hub_enabled is False
    assert s.proxy_max_retries == 5
    assert s.proxy_failure_threshold == 0


def test_env_prefix():
    assert AppSettings.model_config["env_prefix"] == "LLM_"


def test_type_coercion(monkeypatch):
    monkeypatch.setenv("LLM_DEBUG", "true")
    monkeypatch.setenv("LLM_PORT", "9999")
    s = AppSettings()
    assert s.debug is True
    assert s.port == 9999
