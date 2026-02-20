import json
import os
import tempfile
import pytest
from fastapi import HTTPException

from app.services.prompt_manager import StaticPromptStore
from app.schemas.prompt import PromptVariables


@pytest.fixture
def prompt_dir(tmp_path):
    prompt = {
        "input_variables": ["name"],
        "partial_variables": {"greeting": "Hello"},
        "template": "{greeting}, {name}!",
        "metadata": {"category": "test"},
    }
    (tmp_path / "hello.json").write_text(json.dumps(prompt))

    prompt2 = {
        "input_variables": ["x"],
        "template": "Value: {x}",
        "metadata": {"category": "other"},
    }
    (tmp_path / "simple.json").write_text(json.dumps(prompt2))
    return tmp_path


def test_loads_prompts(prompt_dir):
    store = StaticPromptStore(str(prompt_dir))
    assert "hello" in store.stored_prompts
    assert "simple" in store.stored_prompts


@pytest.mark.asyncio
async def test_format_prompt(prompt_dir):
    store = StaticPromptStore(str(prompt_dir))
    result = await store.format_prompt("hello", PromptVariables({"name": "World"}))
    assert result == "Hello, World!"


@pytest.mark.asyncio
async def test_format_prompt_missing_raises_404(prompt_dir):
    store = StaticPromptStore(str(prompt_dir))
    with pytest.raises(HTTPException) as exc_info:
        await store.format_prompt("nonexistent", PromptVariables({}))
    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_format_prompt_missing_vars_raises_400(prompt_dir):
    store = StaticPromptStore(str(prompt_dir))
    with pytest.raises(HTTPException) as exc_info:
        await store.format_prompt("hello", PromptVariables({}))
    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_get_prompts_all(prompt_dir):
    store = StaticPromptStore(str(prompt_dir))
    prompts = await store.get_prompts()
    assert "hello" in prompts
    assert "simple" in prompts


@pytest.mark.asyncio
async def test_get_prompts_filter_by_category(prompt_dir):
    store = StaticPromptStore(str(prompt_dir))
    prompts = await store.get_prompts(category="test")
    assert "hello" in prompts
    assert "simple" not in prompts


def test_missing_directory_raises():
    with pytest.raises(ValueError, match="not found"):
        StaticPromptStore("/nonexistent/path")
