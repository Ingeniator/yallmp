import json
import pytest
from fastapi import HTTPException

from fastapi.responses import JSONResponse

from app.services.chain_manager import (
    _redact_response_headers,
    safe_parse_gigachat_exception,
    StaticChainStore,
)
from app.schemas.prompt import PromptVariables


# --- _redact_response_headers ---

def test_redact_response_headers_dict():
    headers = {"authorization": "Bearer secret-token", "content-type": "application/json"}
    result = _redact_response_headers(headers)
    assert "[REDACTED]" in result["authorization"]
    assert result["content-type"] == "application/json"


def test_redact_response_headers_non_dict():
    assert _redact_response_headers("raw header string") == "[REDACTED]"
    assert _redact_response_headers(b"bytes") == "[REDACTED]"


# --- safe_parse_gigachat_exception ---

def test_safe_parse_valid_args():
    # Double-encoded: the function does json.loads twice for string content
    content = json.dumps(json.dumps({"message": "rate limit exceeded"}))
    exc = Exception("http://api.example.com", 429, content, {"x-request-id": "123"})
    result = safe_parse_gigachat_exception(exc)
    assert result["url"] == "http://api.example.com"
    assert result["status_code"] == 429
    assert result["message"] == "rate limit exceeded"


def test_safe_parse_malformed_args():
    exc = Exception("unexpected single arg")
    result = safe_parse_gigachat_exception(exc)
    assert result["url"] is None
    assert "parse_error" in result
    assert result["message"] == "unexpected single arg"


# --- StaticChainStore ---

@pytest.fixture
def chain_dir(tmp_path):
    chain = {
        "prompt": {
            "input_variables": ["topic"],
            "partial_variables": {},
            "template": "Tell me about {topic}",
        },
        "llm": {"model": "test-model"},
        "metadata": {"description": "test chain", "category": "demo"},
    }
    (tmp_path / "test_chain.json").write_text(json.dumps(chain))
    return tmp_path


def test_get_chains(chain_dir):
    store = StaticChainStore(str(chain_dir))
    import asyncio
    chains = asyncio.get_event_loop().run_until_complete(store.get_chains())
    assert "test_chain" in chains


def test_get_chains_filter(chain_dir):
    store = StaticChainStore(str(chain_dir))
    import asyncio
    chains = asyncio.get_event_loop().run_until_complete(store.get_chains(category="demo"))
    assert "test_chain" in chains
    chains = asyncio.get_event_loop().run_until_complete(store.get_chains(category="nonexistent"))
    assert "test_chain" not in chains


@pytest.mark.asyncio
async def test_execute_missing_chain_raises_404(chain_dir):
    store = StaticChainStore(str(chain_dir))
    with pytest.raises(HTTPException) as exc_info:
        await store.execute("no_such_chain", PromptVariables({}))
    assert exc_info.value.status_code == 404


# --- safe_parse_gigachat_exception: bytes content ---

def test_safe_parse_bytes_content():
    content = json.dumps(json.dumps({"message": "error from server"})).encode("utf-8")
    exc = Exception("http://api.example.com", 500, content, {"content-type": "application/json"})
    result = safe_parse_gigachat_exception(exc)
    assert result["url"] == "http://api.example.com"
    assert result["status_code"] == 500


def test_safe_parse_non_json_content():
    exc = Exception("http://api.example.com", 502, "not json at all", {})
    result = safe_parse_gigachat_exception(exc)
    assert result["url"] == "http://api.example.com"
    assert "raw" in result["message"]


# --- read_config ---

@pytest.mark.asyncio
async def test_read_config(chain_dir):
    store = StaticChainStore(str(chain_dir))
    config = await store.read_config(str(chain_dir / "test_chain.json"))
    assert "prompt" in config
    assert "llm" in config


# --- get_default_available_chat_models ---

@pytest.mark.asyncio
async def test_get_default_available_chat_models(chain_dir):
    store = StaticChainStore(str(chain_dir))
    store.default_available_chat_models = ["ModelA", "ModelB", "ModelC"]

    models = await store.get_default_available_chat_models(exclude="ModelB")
    assert "ModelB" not in models
    assert "ModelA" in models


@pytest.mark.asyncio
async def test_get_default_available_chat_models_no_exclude(chain_dir):
    store = StaticChainStore(str(chain_dir))
    store.default_available_chat_models = ["ModelA", "ModelB"]

    models = await store.get_default_available_chat_models(exclude="NonExistent")
    assert models == ["ModelA", "ModelB"]


# --- malformed chain file ---

def test_malformed_chain_json(tmp_path):
    (tmp_path / "bad.json").write_text("not valid json{{{")
    store = StaticChainStore(str(tmp_path))
    assert "bad" not in store.stored_chains


# --- execute_chain ---

@pytest.mark.asyncio
async def test_execute_chain_success(chain_dir):
    from unittest.mock import AsyncMock, MagicMock, patch
    store = StaticChainStore(str(chain_dir))
    store.default_available_chat_models = ["ModelA"]

    mock_llm = MagicMock()
    mock_llm.timeout = 600
    mock_llm.base_url = "http://llm"
    mock_llm.model = "test-model"
    mock_llm.ca_bundle_file = ""
    mock_llm.cert_file = ""
    mock_llm.key_file = ""
    mock_llm.auth_url = ""

    mock_chain = MagicMock()
    mock_chain.llm = mock_llm
    mock_chain.ainvoke = AsyncMock(return_value={"output": "hello"})

    with patch("app.services.chain_manager.settings") as s:
        s.timeout_keep_alive = 600
        s.chain_default_base_url = ""
        s.chain_default_model_name = ""
        s.chain_default_ca_bundle_file = ""
        s.chain_default_cert_file = ""
        s.chain_default_key_file = ""
        s.chain_default_auth_url = ""
        s.chain_default_credentials = ""
        s.chain_default_scope = ""

        result = await store.execute_chain(chain=mock_chain, variables={"topic": "test"})

    assert result == {"output": "hello"}
    mock_chain.ainvoke.assert_awaited_once()


@pytest.mark.asyncio
async def test_execute_chain_error(chain_dir):
    from unittest.mock import AsyncMock, MagicMock, patch
    store = StaticChainStore(str(chain_dir))
    store.default_available_chat_models = ["ModelA"]

    mock_llm = MagicMock()
    mock_llm.timeout = 600
    mock_llm.base_url = "http://llm"
    mock_llm.model = "test-model"
    mock_llm.ca_bundle_file = ""
    mock_llm.cert_file = ""
    mock_llm.key_file = ""
    mock_llm.auth_url = ""

    mock_chain = MagicMock()
    mock_chain.llm = mock_llm
    mock_chain.ainvoke = AsyncMock(side_effect=Exception("http://api", 500, "error body", {}))

    with patch("app.services.chain_manager.settings") as s:
        s.timeout_keep_alive = 600
        s.chain_default_base_url = ""
        s.chain_default_model_name = ""
        s.chain_default_ca_bundle_file = ""
        s.chain_default_cert_file = ""
        s.chain_default_key_file = ""
        s.chain_default_auth_url = ""
        s.chain_default_credentials = ""
        s.chain_default_scope = ""

        result = await store.execute_chain(chain=mock_chain, variables={})

    assert isinstance(result, JSONResponse)
    assert result.status_code == 500


@pytest.mark.asyncio
async def test_execute_chain_model_override(chain_dir):
    """When model_name is provided, chain.llm.model is overridden."""
    from unittest.mock import AsyncMock, MagicMock, patch
    store = StaticChainStore(str(chain_dir))
    store.default_available_chat_models = []

    mock_llm = MagicMock()
    mock_llm.timeout = 600
    mock_llm.base_url = "http://llm"
    mock_llm.model = "original-model"
    mock_llm.ca_bundle_file = ""
    mock_llm.cert_file = ""
    mock_llm.key_file = ""
    mock_llm.auth_url = ""

    mock_chain = MagicMock()
    mock_chain.llm = mock_llm
    mock_chain.ainvoke = AsyncMock(return_value={"output": "ok"})

    with patch("app.services.chain_manager.settings") as s:
        s.timeout_keep_alive = 600
        s.chain_default_base_url = ""
        s.chain_default_model_name = ""
        s.chain_default_ca_bundle_file = ""
        s.chain_default_cert_file = ""
        s.chain_default_key_file = ""
        s.chain_default_auth_url = ""
        s.chain_default_credentials = ""
        s.chain_default_scope = ""

        await store.execute_chain(chain=mock_chain, model_name="override-model")

    assert mock_chain.llm.model == "override-model"


@pytest.mark.asyncio
async def test_execute_chain_config_defaults_applied(chain_dir):
    """When chain.llm fields are empty, defaults from settings are applied."""
    from unittest.mock import AsyncMock, MagicMock, patch
    store = StaticChainStore(str(chain_dir))
    store.default_available_chat_models = []

    mock_llm = MagicMock()
    mock_llm.timeout = None
    mock_llm.base_url = ""
    mock_llm.model = ""
    mock_llm.ca_bundle_file = ""
    mock_llm.cert_file = ""
    mock_llm.key_file = ""
    mock_llm.auth_url = ""

    mock_chain = MagicMock()
    mock_chain.llm = mock_llm
    mock_chain.ainvoke = AsyncMock(return_value={"output": "ok"})

    with patch("app.services.chain_manager.settings") as s:
        s.timeout_keep_alive = 300
        s.chain_default_base_url = "http://default-llm"
        s.chain_default_model_name = "default-model"
        s.chain_default_ca_bundle_file = "/ca.pem"
        s.chain_default_cert_file = "/cert.pem"
        s.chain_default_key_file = "/key.pem"
        s.chain_default_auth_url = "http://auth"
        s.chain_default_credentials = "creds"
        s.chain_default_scope = "SCOPE"

        await store.execute_chain(chain=mock_chain)

    assert mock_llm.timeout == 300
    assert mock_llm.base_url == "http://default-llm"
    assert mock_llm.model == "default-model"
    assert mock_llm.ca_bundle_file == "/ca.pem"
    assert mock_llm.cert_file == "/cert.pem"
    assert mock_llm.key_file == "/key.pem"
    assert mock_llm.auth_url == "http://auth"


def test_missing_directory_raises():
    with pytest.raises(ValueError, match="not found"):
        StaticChainStore("/nonexistent/path")
