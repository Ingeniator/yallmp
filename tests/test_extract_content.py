import json
import pytest
from fastapi.responses import JSONResponse
from httpx import Response as HTTPXResponse, Request as HTTPXRequest

from app.core.proxy import extract_content


def _make_httpx_response(status_code=200, json_data=None, text=None):
    kwargs = {"status_code": status_code, "request": HTTPXRequest("GET", "http://test")}
    if json_data is not None:
        kwargs["json"] = json_data
    elif text is not None:
        kwargs["text"] = text
    return HTTPXResponse(**kwargs)


def test_httpx_response_with_json():
    resp = _make_httpx_response(json_data={"ok": True})
    assert extract_content(resp) == {"ok": True}


def test_httpx_response_with_text():
    resp = _make_httpx_response(text="plain text body")
    assert extract_content(resp) == {"detail": "plain text body"}


def test_json_response():
    resp = JSONResponse(content={"result": 42}, status_code=200)
    assert extract_content(resp) == {"result": 42}


def test_httpx_response_raise_exception():
    """When raiseException=True and both json() and text fail, exception propagates."""
    from unittest.mock import PropertyMock, patch

    resp = HTTPXResponse(
        status_code=200,
        content=b"not json",
        request=HTTPXRequest("GET", "http://test"),
    )
    # .json() fails (not valid JSON), then patch .text to also raise
    with patch.object(HTTPXResponse, "text", new_callable=PropertyMock, side_effect=RuntimeError("decode fail")):
        with pytest.raises(RuntimeError, match="decode fail"):
            extract_content(resp, raiseException=True)


def test_json_response_raise_exception():
    """When raiseException=True and body parsing fails, exception propagates."""
    resp = JSONResponse(content={"a": 1})
    resp.body = b"not json"

    with pytest.raises(Exception):
        extract_content(resp, raiseException=True)


def test_unknown_type_returns_fallback():
    result = extract_content("not a response")
    assert result == {"detail": "Unknown response object type"}
