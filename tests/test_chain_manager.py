import json
import pytest
from fastapi import HTTPException

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


def test_missing_directory_raises():
    with pytest.raises(ValueError, match="not found"):
        StaticChainStore("/nonexistent/path")
