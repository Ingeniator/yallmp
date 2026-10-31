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


def test_proxy_authorization_type_default_is_oidc(monkeypatch):
    monkeypatch.delenv("LLM_PROXY_AUTHORIZATION_TYPE", raising=False)
    assert AppSettings(_env_file=None).proxy_authorization_type == "OIDC"


def test_proxy_authorization_type_bearer_is_static_key(monkeypatch):
    monkeypatch.setenv("LLM_PROXY_AUTHORIZATION_TYPE", "bearer")
    monkeypatch.setenv("LLM_PROXY_API_KEY", "k")
    monkeypatch.delenv("LLM_PROXY_OIDC_AUTHORIZATION_URL", raising=False)
    s = AppSettings(_env_file=None)
    assert s.proxy_authorization_type == "BEARER"


def test_proxy_authorization_type_legacy_bearer_with_oidc_url_becomes_oidc(monkeypatch):
    import pytest

    monkeypatch.setenv("LLM_PROXY_AUTHORIZATION_TYPE", "BEARER")
    monkeypatch.setenv("LLM_PROXY_OIDC_AUTHORIZATION_URL", "http://auth/token")
    monkeypatch.delenv("LLM_PROXY_API_KEY", raising=False)
    with pytest.warns(DeprecationWarning, match="use OIDC"):
        s = AppSettings(_env_file=None)
    assert s.proxy_authorization_type == "OIDC"


def test_proxy_authorization_type_is_upper_cased(monkeypatch):
    monkeypatch.setenv("LLM_PROXY_AUTHORIZATION_TYPE", "apikey")
    assert AppSettings(_env_file=None).proxy_authorization_type == "APIKEY"


def test_proxy_authorization_type_rejects_unknown_value(monkeypatch):
    import pytest
    from pydantic import ValidationError

    monkeypatch.setenv("LLM_PROXY_AUTHORIZATION_TYPE", "OIDC|CERT|APIKEY|NONE")
    with pytest.raises(ValidationError, match="proxy_authorization_type"):
        AppSettings(_env_file=None)
