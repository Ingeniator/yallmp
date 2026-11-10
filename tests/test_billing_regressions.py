"""Regression tests for billing leaks:

1. charge() must write the same (sanitized) keys BillingMiddleware reads.
2. Streaming chat completions must request usage so they can be charged,
   without leaking the extra usage-only chunk to clients that did not ask for it.
"""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.core.proxy import (
    _drop_usage_only_events,
    _ensure_stream_usage,
    proxy_request_to_provider,
)
from app.core.security import sanitize_group_id
from app.services.billing import charge, period_key
from tests.test_proxy_multi_provider import FakeProvider, _make_request

_LIMITS = {
    "tiers": {
        "tier1": {"period": "month", "group_limit": 100.0, "user_limit": 10.0, "alert_threshold": 0.8}
    },
    "orgs": {"default": "tier1"},
}


class MemRedis:
    """Just enough of redis.asyncio for charge()."""

    def __init__(self):
        self.data: dict[str, float] = {}

    async def incrbyfloat(self, key, amount):
        self.data[key] = self.data.get(key, 0.0) + amount
        return self.data[key]

    async def expire(self, *args, **kwargs):
        return True

    async def get(self, key):
        return self.data.get(key)


# ── 1. charge / middleware key consistency ────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize("raw", [" acme/alice ", "acme//alice", "/acme/alice/", "acme/alice"])
async def test_charge_uses_same_keys_as_middleware(raw):
    redis = MemRedis()
    await charge(redis, _LIMITS, raw, 5.0)

    # BillingMiddleware derives its keys from sanitize_group_id(header).
    group_id = sanitize_group_id(raw)
    org = group_id.split("/")[0]
    pk = period_key("month")
    assert await redis.get(f"billing:group:{org}:{pk}") == 5.0
    assert await redis.get(f"billing:user:{group_id}:{pk}") == 5.0
    assert set(redis.data) == {f"billing:group:{org}:{pk}", f"billing:user:{group_id}:{pk}"}


@pytest.mark.asyncio
async def test_charge_empty_group_id_goes_to_unknown():
    redis = MemRedis()
    await charge(redis, _LIMITS, "", 1.0)
    assert set(redis.data) == {f"billing:group:unknown:{period_key('month')}"}


# ── 2a. _ensure_stream_usage ──────────────────────────────────────────────────

_PATH = "v1/chat/completions"


def _body(**kw) -> bytes:
    return json.dumps({"model": "m", "messages": [], **kw}).encode()


def test_ensure_stream_usage_injects_when_missing():
    body, injected = _ensure_stream_usage(_body(stream=True), _PATH)
    assert injected
    assert json.loads(body)["stream_options"] == {"include_usage": True}


def test_ensure_stream_usage_overrides_false_and_keeps_other_options():
    body, injected = _ensure_stream_usage(
        _body(stream=True, stream_options={"include_usage": False, "x": 1}), _PATH
    )
    assert injected
    assert json.loads(body)["stream_options"] == {"include_usage": True, "x": 1}


def test_ensure_stream_usage_respects_client_request():
    original = _body(stream=True, stream_options={"include_usage": True})
    body, injected = _ensure_stream_usage(original, _PATH)
    assert not injected
    assert body == original


@pytest.mark.parametrize(
    "body,path",
    [
        (_body(), _PATH),  # not streaming
        (_body(stream=False), _PATH),
        (_body(stream=True), "v1/responses"),  # Responses API reports usage itself
        (_body(stream=True), "v1/embeddings"),
        (b"not json", _PATH),
        (b"", _PATH),
    ],
)
def test_ensure_stream_usage_noop(body, path):
    assert _ensure_stream_usage(body, path) == (body, False)


# ── 2b. _drop_usage_only_events ───────────────────────────────────────────────

_CONTENT = 'data: {"choices":[{"delta":{"content":"при"}}],"usage":null}\n\n'
_USAGE = 'data: {"choices":[],"usage":{"prompt_tokens":3,"completion_tokens":5}}\n\n'
_DONE = "data: [DONE]\n\n"


async def _collect(chunks: list[bytes]) -> str:
    async def gen():
        for c in chunks:
            yield c

    return b"".join([c async for c in _drop_usage_only_events(gen())]).decode()


@pytest.mark.asyncio
async def test_drop_usage_only_events_removes_usage_chunk():
    out = await _collect([(_CONTENT + _USAGE + _DONE).encode()])
    assert out == _CONTENT + _DONE


@pytest.mark.asyncio
async def test_drop_usage_only_events_handles_split_chunks_and_multibyte():
    raw = (_CONTENT + _USAGE + _DONE).encode()
    # split every 7 bytes: breaks events and Cyrillic multibyte chars mid-way
    out = await _collect([raw[i : i + 7] for i in range(0, len(raw), 7)])
    assert out == _CONTENT + _DONE


@pytest.mark.asyncio
async def test_drop_usage_only_events_keeps_unterminated_tail():
    out = await _collect([(_CONTENT + "data: partial").encode()])
    assert out == _CONTENT + "data: partial"


# ── 2c. End-to-end through proxy_request_to_provider ──────────────────────────


def _streaming_upstream(payload: bytes):
    resp = MagicMock()
    resp.status_code = 200

    async def aiter_bytes():
        yield payload

    resp.aiter_bytes = aiter_bytes
    resp.aclose = AsyncMock()
    return resp


async def _run_stream(provider, body: bytes):
    upstream = _streaming_upstream((_CONTENT + _USAGE + _DONE).encode())
    provider.client.send = AsyncMock(return_value=upstream)
    provider.client.build_request = MagicMock(return_value="built")

    with patch("app.core.proxy.settings") as s, patch(
        "app.core.proxy._emit_streaming_metrics"
    ) as emit:
        s.proxy_exclude_headers = "host"
        resp = await proxy_request_to_provider(
            provider=provider,
            path=_PATH,
            request=_make_request(body=body),
            auth_headers={},
            original_model="test/m",
            stripped_model="m",
        )
        client_out = b"".join([c async for c in resp.body_iterator]).decode()

    sent_body = json.loads(provider.client.build_request.call_args.kwargs["content"])
    metrics_text = "".join(emit.call_args.args[0])
    return sent_body, client_out, metrics_text


@pytest.mark.asyncio
async def test_streaming_requests_usage_and_hides_it_from_client():
    sent, client_out, metrics_text = await _run_stream(
        FakeProvider(), json.dumps({"model": "test/m", "stream": True}).encode()
    )
    assert sent["stream_options"] == {"include_usage": True}
    assert '"prompt_tokens":3' in metrics_text  # billed
    assert '"prompt_tokens"' not in client_out  # not leaked
    assert client_out == _CONTENT + _DONE


@pytest.mark.asyncio
async def test_streaming_forwards_usage_when_client_asked():
    _, client_out, metrics_text = await _run_stream(
        FakeProvider(),
        json.dumps(
            {"model": "test/m", "stream": True, "stream_options": {"include_usage": True}}
        ).encode(),
    )
    assert client_out == _CONTENT + _USAGE + _DONE
    assert '"prompt_tokens":3' in metrics_text


@pytest.mark.asyncio
async def test_streaming_injection_can_be_disabled_per_provider():
    provider = FakeProvider()
    provider.config.inject_stream_usage = False
    sent, client_out, _ = await _run_stream(
        provider, json.dumps({"model": "test/m", "stream": True}).encode()
    )
    assert "stream_options" not in sent
    assert client_out == _CONTENT + _USAGE + _DONE  # passthrough untouched
