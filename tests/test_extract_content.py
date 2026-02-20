import json
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


def test_unknown_type_returns_fallback():
    result = extract_content("not a response")
    assert result == {"detail": "Unknown response object type"}
