from app.core.config import AppSettings


def test_default_values(monkeypatch):
    # Clear env vars that the local .env may set so we get true defaults
    for key in ("LLM_DEBUG", "LLM_PORT", "LLM_PROXY_ENABLED", "LLM_PROXY_TARGET_URL",
                "LLM_PROXY_AUTHORIZATION_TYPE", "LLM_PROXY_PRICING_CONFIG",
                "LLM_DASHBOARD_ENABLED", "LLM_SEARCH_HUB_ENABLED", "EXA_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    s = AppSettings(_env_file=None)
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


def test_llm_hub_enabled_is_alias_of_proxy_enabled(monkeypatch):
    import pytest

    monkeypatch.delenv("LLM_PROXY_ENABLED", raising=False)
    monkeypatch.setenv("LLM_LLM_HUB_ENABLED", "true")
    with pytest.warns(DeprecationWarning, match="use proxy_enabled"):
        s = AppSettings(_env_file=None)
    assert s.proxy_enabled is True


def test_removed_proxy_settings_detected(monkeypatch):
    from app.core.config import removed_proxy_settings_in_use

    monkeypatch.setenv("LLM_PROXY_TARGET_URL", "http://old")
    monkeypatch.delenv("LLM_PROXY_AUTHORIZATION_TYPE", raising=False)
    s = AppSettings(_env_file=None)
    assert "proxy_target_url" in removed_proxy_settings_in_use(s)
    assert "proxy_authorization_type" not in removed_proxy_settings_in_use(s)


def test_removed_proxy_settings_detected_in_env_file(monkeypatch, tmp_path):
    from app.core.config import removed_proxy_settings_in_use

    monkeypatch.delenv("LLM_PROXY_TARGET_URL", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text("LLM_PROXY_TARGET_URL=http://old\n")
    s = AppSettings(_env_file=str(env_file))
    assert removed_proxy_settings_in_use(s) == ["proxy_target_url"]
