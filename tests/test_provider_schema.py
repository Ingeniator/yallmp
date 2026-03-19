import pytest
from app.schemas.provider import AuthType, AuthConfig, TimeoutConfig, LlmProviderConfig


def test_auth_type_enum_values():
    assert AuthType.BEARER == "BEARER"
    assert AuthType.APIKEY == "APIKEY"
    assert AuthType.CERT == "CERT"
    assert AuthType.NONE == "NONE"


def test_auth_config_defaults():
    auth = AuthConfig()
    assert auth.type == AuthType.NONE
    assert auth.oidc_url is None
    assert auth.credentials is None
    assert auth.scope is None
    assert auth.api_key is None


def test_auth_config_bearer():
    auth = AuthConfig(type=AuthType.BEARER, oidc_url="http://auth/token", credentials="abc", scope="SCOPE")
    assert auth.type == AuthType.BEARER
    assert auth.oidc_url == "http://auth/token"
    assert auth.scope == "SCOPE"


def test_timeout_config_defaults():
    t = TimeoutConfig()
    assert t.connect == 10
    assert t.read == 300
    assert t.write == 30
    assert t.pool is None


def test_timeout_config_custom():
    t = TimeoutConfig(connect=5, read=60, write=10, pool=20)
    assert t.connect == 5
    assert t.pool == 20


def test_provider_config_minimal():
    config = LlmProviderConfig(prefix="test", base_url="http://test.com")
    assert config.prefix == "test"
    assert config.base_url == "http://test.com"
    assert config.models == []
    assert config.verify_ssl is True
    assert config.auth.type == AuthType.NONE
    assert config.timeout.connect == 10
    assert config.failure_threshold == 0
    assert config.max_retries == 5
    assert config.base_delay == 0.5
    assert config.backoff_factor == 2.0


def test_provider_config_full():
    data = {
        "prefix": "gigachat",
        "base_url": "https://gigachat.api/v1",
        "auth": {
            "type": "BEARER",
            "oidc_url": "http://auth/token",
            "credentials": "creds",
            "scope": "GIGACHAT_API_CORP",
        },
        "models": ["GigaChat-2:latest", "DeepSeek-R1"],
        "verify_ssl": False,
        "timeout": {"connect": 5, "read": 120, "write": 15},
        "failure_threshold": 3,
        "max_retries": 10,
        "base_delay": 1.0,
        "backoff_factor": 3.0,
    }
    config = LlmProviderConfig(**data)
    assert config.prefix == "gigachat"
    assert config.auth.type == AuthType.BEARER
    assert config.auth.scope == "GIGACHAT_API_CORP"
    assert len(config.models) == 2
    assert config.verify_ssl is False
    assert config.timeout.read == 120
    assert config.failure_threshold == 3
    assert config.max_retries == 10


def test_provider_config_from_json_string():
    import json
    raw = '{"prefix": "p", "base_url": "http://x", "models": ["m1"]}'
    config = LlmProviderConfig(**json.loads(raw))
    assert config.prefix == "p"
    assert config.models == ["m1"]


def test_provider_config_missing_required_fields():
    with pytest.raises(Exception):
        LlmProviderConfig(prefix="x")  # missing base_url

    with pytest.raises(Exception):
        LlmProviderConfig(base_url="http://x")  # missing prefix
