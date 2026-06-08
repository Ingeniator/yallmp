"""End-to-end tests for the billing sync flow.

Covers: sync_from_llogr (happy path, pagination, GT semantics, error handling,
lock behaviour) and billing_sync_loop (periodic scheduling).
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest


# ── Shared fixtures / helpers ─────────────────────────────────────────────────

_LIMITS = {
    "tiers": {
        "tier1": {
            "period": "month",
            "group_limit": 100.0,
            "user_limit": 10.0,
            "alert_threshold": 0.8,
        },
    },
    "orgs": {"acme": "tier1", "globex": "tier1", "default": "tier1"},
}

_LIMITS_MIXED_PERIODS = {
    "tiers": {
        "monthly": {"period": "month", "group_limit": 100.0, "user_limit": 10.0, "alert_threshold": 0.8},
        "weekly":  {"period": "week",  "group_limit":  25.0, "user_limit":  5.0, "alert_threshold": 0.8},
    },
    "orgs": {"acme": "monthly", "beta": "weekly", "default": "monthly"},
}

LLOGR_URL = "http://llogr:5000"


@pytest.fixture(autouse=True)
def fresh_lock():
    """Replace the module-level lock before each test so tests are isolated."""
    from app.services import billing_sync
    billing_sync._sync_lock = asyncio.Lock()


@pytest.fixture
def pipe():
    p = MagicMock()
    p.execute = AsyncMock(return_value=[])
    return p


@pytest.fixture
def redis(pipe):
    r = MagicMock()
    r.pipeline.return_value = pipe
    return r


def _page(groups=(), users=(), has_more=False):
    """Build a minimal llogr billing summary page."""
    return {"groups": list(groups), "users": list(users), "has_more": has_more}


def _mock_http(*pages):
    """Patch httpx.AsyncClient so successive GET calls return the given pages.

    Returns (patcher, mock_client).  The patcher is used as a context manager.
    """
    responses = []
    for p in pages:
        m = MagicMock()
        m.json.return_value = p
        m.raise_for_status = MagicMock()
        responses.append(m)

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(side_effect=responses)

    mock_acm = MagicMock()
    mock_acm.__aenter__ = AsyncMock(return_value=mock_client)
    mock_acm.__aexit__ = AsyncMock(return_value=None)

    patcher = patch("app.services.billing_sync.httpx.AsyncClient", return_value=mock_acm)
    return patcher, mock_client


# ── Happy path ────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_sync_writes_group_and_user_keys(redis, pipe):
    from app.services.billing_sync import sync_from_llogr

    page = _page(
        groups=[
            {"org": "acme",   "group_spent": 42.318450},
            {"org": "globex", "group_spent": 95.0},
        ],
        users=[
            {"project_id": "acme/alice", "user_spent": 8.75},
            {"project_id": "acme/bob",   "user_spent": 3.14},
        ],
    )

    patcher, _ = _mock_http(page)
    with patcher:
        await sync_from_llogr(redis, _LIMITS, LLOGR_URL)

    written = [c.args[0] for c in pipe.set.call_args_list]
    assert any("billing:group:acme:"   in k for k in written)
    assert any("billing:group:globex:" in k for k in written)
    assert any("billing:user:acme/alice:" in k for k in written)
    assert any("billing:user:acme/bob:"   in k for k in written)
    pipe.execute.assert_awaited_once()


@pytest.mark.asyncio
async def test_sync_writes_correct_spent_values(redis, pipe):
    from app.services.billing_sync import sync_from_llogr
    from app.services.billing import period_key

    page = _page(
        groups=[{"org": "acme", "group_spent": 42.318450}],
        users=[{"project_id": "acme/alice", "user_spent": 8.75}],
    )

    patcher, _ = _mock_http(page)
    with patcher:
        await sync_from_llogr(redis, _LIMITS, LLOGR_URL)

    pk = period_key("month")
    kv = {c.args[0]: c.args[1] for c in pipe.set.call_args_list}
    assert kv[f"billing:group:acme:{pk}"]      == 42.318450
    assert kv[f"billing:user:acme/alice:{pk}"] == 8.75


@pytest.mark.asyncio
async def test_sync_uses_gt_and_ttl_on_every_set(redis, pipe):
    from app.services.billing_sync import sync_from_llogr

    page = _page(
        groups=[{"org": "acme", "group_spent": 10.0}],
        users=[{"project_id": "acme/alice", "user_spent": 2.0}],
    )

    patcher, _ = _mock_http(page)
    with patcher:
        await sync_from_llogr(redis, _LIMITS, LLOGR_URL)

    assert pipe.set.call_count == 2
    for c in pipe.set.call_args_list:
        assert c.kwargs.get("gt") is True,       f"gt=True missing on SET: {c}"
        assert c.kwargs.get("ex") is not None,   f"TTL (ex=) missing on SET: {c}"


@pytest.mark.asyncio
async def test_sync_calls_llogr_with_correct_url_and_headers(redis, pipe):
    from app.services.billing_sync import sync_from_llogr

    patcher, mock_client = _mock_http(_page())
    with patcher:
        await sync_from_llogr(redis, _LIMITS, LLOGR_URL)

    mock_client.get.assert_awaited_once()
    call_kwargs = mock_client.get.call_args
    assert call_kwargs.args[0] == f"{LLOGR_URL}/api/public/billing/summary"
    assert call_kwargs.kwargs["headers"]["X-Role"] == "SUPER_ADMIN"
    assert "start" in call_kwargs.kwargs["params"]
    assert "end"   in call_kwargs.kwargs["params"]


# ── Pagination ────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_sync_follows_has_more_flag(redis, pipe):
    from app.services.billing_sync import sync_from_llogr

    page1 = _page(
        groups=[{"org": "acme", "group_spent": 50.0}],
        users=[{"project_id": "acme/alice", "user_spent": 8.0}],
        has_more=True,
    )
    page2 = _page(
        groups=[{"org": "acme", "group_spent": 50.0}],  # groups returned again on page 2
        users=[{"project_id": "acme/bob", "user_spent": 3.0}],
        has_more=False,
    )

    patcher, mock_client = _mock_http(page1, page2)
    with patcher:
        await sync_from_llogr(redis, _LIMITS, LLOGR_URL)

    assert mock_client.get.await_count == 2

    written = [c.args[0] for c in pipe.set.call_args_list]
    assert any("acme/alice" in k for k in written), "page-1 user missing"
    assert any("acme/bob"   in k for k in written), "page-2 user missing"

    # Groups must be written exactly once even though they appear on both pages
    group_writes = [k for k in written if "billing:group:acme:" in k]
    assert len(group_writes) == 1


@pytest.mark.asyncio
async def test_sync_passes_correct_offset_per_page(redis, pipe):
    from app.services.billing_sync import sync_from_llogr, _PAGE_SIZE

    page1 = _page(users=[{"project_id": "acme/u1", "user_spent": 1.0}], has_more=True)
    page2 = _page(users=[{"project_id": "acme/u2", "user_spent": 1.0}], has_more=False)

    patcher, mock_client = _mock_http(page1, page2)
    with patcher:
        await sync_from_llogr(redis, _LIMITS, LLOGR_URL)

    offsets = [c.kwargs["params"]["user_offset"] for c in mock_client.get.call_args_list]
    assert offsets == [0, _PAGE_SIZE]


@pytest.mark.asyncio
async def test_sync_all_pages_flushed_in_one_pipeline_execute(redis, pipe):
    from app.services.billing_sync import sync_from_llogr

    page1 = _page(users=[{"project_id": "acme/u1", "user_spent": 1.0}], has_more=True)
    page2 = _page(users=[{"project_id": "acme/u2", "user_spent": 1.0}], has_more=False)

    patcher, _ = _mock_http(page1, page2)
    with patcher:
        await sync_from_llogr(redis, _LIMITS, LLOGR_URL)

    # All SET commands accumulated across pages, then flushed in a single execute()
    pipe.execute.assert_awaited_once()
    assert pipe.set.call_count == 2  # one per user


# ── Lock / concurrency ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_sync_skips_when_lock_already_held(redis, pipe):
    from app.services import billing_sync

    # Hold the lock in a background coroutine
    async def _hold():
        async with billing_sync._sync_lock:
            await asyncio.sleep(10)

    holder = asyncio.create_task(_hold())
    await asyncio.sleep(0)  # yield so holder acquires the lock first
    assert billing_sync._sync_lock.locked()

    patcher, mock_client = _mock_http()
    with patcher:
        await billing_sync.sync_from_llogr(redis, _LIMITS, LLOGR_URL)

    mock_client.get.assert_not_called()
    pipe.set.assert_not_called()

    holder.cancel()
    try:
        await holder
    except asyncio.CancelledError:
        pass


@pytest.mark.asyncio
async def test_lock_released_after_successful_sync(redis, pipe):
    from app.services import billing_sync

    patcher, _ = _mock_http(_page())
    with patcher:
        await billing_sync.sync_from_llogr(redis, _LIMITS, LLOGR_URL)

    assert not billing_sync._sync_lock.locked()


@pytest.mark.asyncio
async def test_lock_released_after_fetch_error(redis, pipe):
    from app.services import billing_sync

    mock_client = AsyncMock()
    mock_client.get.side_effect = Exception("connection refused")
    mock_acm = MagicMock()
    mock_acm.__aenter__ = AsyncMock(return_value=mock_client)
    mock_acm.__aexit__ = AsyncMock(return_value=None)

    with patch("app.services.billing_sync.httpx.AsyncClient", return_value=mock_acm):
        await billing_sync.sync_from_llogr(redis, _LIMITS, LLOGR_URL)

    assert not billing_sync._sync_lock.locked()


# ── Error handling ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_sync_survives_llogr_http_error(redis, pipe):
    from app.services.billing_sync import sync_from_llogr

    mock_client = AsyncMock()
    mock_client.get.side_effect = Exception("connection refused")
    mock_acm = MagicMock()
    mock_acm.__aenter__ = AsyncMock(return_value=mock_client)
    mock_acm.__aexit__ = AsyncMock(return_value=None)

    with patch("app.services.billing_sync.httpx.AsyncClient", return_value=mock_acm):
        await sync_from_llogr(redis, _LIMITS, LLOGR_URL)  # must not raise

    pipe.set.assert_not_called()


@pytest.mark.asyncio
async def test_sync_survives_redis_pipeline_error(redis, pipe):
    from app.services.billing_sync import sync_from_llogr

    pipe.execute = AsyncMock(side_effect=ConnectionError("redis down"))
    page = _page(groups=[{"org": "acme", "group_spent": 10.0}])

    patcher, _ = _mock_http(page)
    with patcher:
        await sync_from_llogr(redis, _LIMITS, LLOGR_URL)  # must not raise


@pytest.mark.asyncio
async def test_sync_survives_llogr_error_mid_pagination(redis, pipe):
    from app.services.billing_sync import sync_from_llogr

    # Page 1 succeeds, page 2 fails — should write page 1 data only
    page1 = _page(users=[{"project_id": "acme/alice", "user_spent": 5.0}], has_more=True)
    ok_response = MagicMock()
    ok_response.json.return_value = page1
    ok_response.raise_for_status = MagicMock()

    fail_response = MagicMock()
    fail_response.raise_for_status.side_effect = Exception("timeout")

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(side_effect=[ok_response, fail_response])
    mock_acm = MagicMock()
    mock_acm.__aenter__ = AsyncMock(return_value=mock_client)
    mock_acm.__aexit__ = AsyncMock(return_value=None)

    with patch("app.services.billing_sync.httpx.AsyncClient", return_value=mock_acm):
        await sync_from_llogr(redis, _LIMITS, LLOGR_URL)  # must not raise

    # Page 1 user should have been accumulated before the failure
    written = [c.args[0] for c in pipe.set.call_args_list]
    assert any("acme/alice" in k for k in written)


# ── Data filtering ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_sync_skips_users_without_slash(redis, pipe):
    from app.services.billing_sync import sync_from_llogr

    page = _page(users=[
        {"project_id": "acme",       "user_spent": 5.0},  # org-only — skip
        {"project_id": "",            "user_spent": 1.0},  # empty — skip
        {"project_id": "acme/alice", "user_spent": 3.0},  # valid
    ])

    patcher, _ = _mock_http(page)
    with patcher:
        await sync_from_llogr(redis, _LIMITS, LLOGR_URL)

    written = [c.args[0] for c in pipe.set.call_args_list]
    user_keys = [k for k in written if "billing:user:" in k]

    assert len(user_keys) == 1
    assert "acme/alice" in user_keys[0]


@pytest.mark.asyncio
async def test_sync_skips_groups_with_empty_org(redis, pipe):
    from app.services.billing_sync import sync_from_llogr

    page = _page(groups=[
        {"org": "",     "group_spent": 5.0},  # skip
        {"org": "acme", "group_spent": 42.0}, # valid
    ])

    patcher, _ = _mock_http(page)
    with patcher:
        await sync_from_llogr(redis, _LIMITS, LLOGR_URL)

    written = [c.args[0] for c in pipe.set.call_args_list]
    group_keys = [k for k in written if "billing:group:" in k]
    assert len(group_keys) == 1
    assert "acme" in group_keys[0]


@pytest.mark.asyncio
async def test_sync_handles_none_spent_values(redis, pipe):
    from app.services.billing_sync import sync_from_llogr

    # ClickHouse may return None for aggregations on no rows (edge case)
    page = _page(
        groups=[{"org": "acme", "group_spent": None}],
        users=[{"project_id": "acme/alice", "user_spent": None}],
    )

    patcher, _ = _mock_http(page)
    with patcher:
        await sync_from_llogr(redis, _LIMITS, LLOGR_URL)  # must not raise

    kv = {c.args[0]: c.args[1] for c in pipe.set.call_args_list}
    assert all(v == 0.0 for v in kv.values())


# ── Multiple period types ─────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_sync_fetches_separate_window_per_period_type(redis, pipe):
    from app.services.billing_sync import sync_from_llogr

    # _LIMITS_MIXED_PERIODS has both "month" and "week" tiers — expect 2 HTTP calls
    patcher, mock_client = _mock_http(_page(), _page())
    with patcher:
        await sync_from_llogr(redis, _LIMITS_MIXED_PERIODS, LLOGR_URL)

    assert mock_client.get.await_count == 2

    # The two calls must use distinct (start, end) pairs.
    # We compare full param dicts rather than individual fields to avoid false
    # failures on the rare calendar day where month-start == week-start.
    p0 = mock_client.get.call_args_list[0].kwargs["params"]
    p1 = mock_client.get.call_args_list[1].kwargs["params"]
    assert (p0["start"], p0["end"]) != (p1["start"], p1["end"]), (
        "month and week windows must differ in at least start or end"
    )


@pytest.mark.asyncio
async def test_sync_uses_correct_period_key_per_org_tier(redis, pipe):
    from app.services.billing_sync import sync_from_llogr
    from app.services.billing import period_key

    # "acme" → monthly tier, "beta" → weekly tier
    page_month = _page(groups=[{"org": "acme", "group_spent": 10.0}])
    page_week  = _page(groups=[{"org": "beta", "group_spent": 5.0}])

    patcher, _ = _mock_http(page_month, page_week)
    with patcher:
        await sync_from_llogr(redis, _LIMITS_MIXED_PERIODS, LLOGR_URL)

    written_keys = [c.args[0] for c in pipe.set.call_args_list]
    assert any(f"billing:group:acme:{period_key('month')}" in k for k in written_keys)
    assert any(f"billing:group:beta:{period_key('week')}"  in k for k in written_keys)


# ── Loop ─────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_billing_sync_loop_calls_sync_repeatedly():
    from app.services.billing_sync import billing_sync_loop

    redis  = MagicMock()
    limits = _LIMITS
    sync_calls: list[str] = []
    sleep_calls: list[int] = []

    async def _fake_sync(r, l, url):
        sync_calls.append(url)

    async def _fake_sleep(n):
        sleep_calls.append(n)
        if len(sleep_calls) >= 3:
            raise asyncio.CancelledError

    with patch("app.services.billing_sync.sync_from_llogr", side_effect=_fake_sync), \
         patch("asyncio.sleep", side_effect=_fake_sleep):
        try:
            await billing_sync_loop(redis, limits, LLOGR_URL, interval=60)
        except asyncio.CancelledError:
            pass

    # sleep fires, then sync: 2 completed cycles before 3rd sleep raises
    assert len(sync_calls) == 2
    assert all(s == 60 for s in sleep_calls)
    assert all(u == LLOGR_URL for u in sync_calls)


@pytest.mark.asyncio
async def test_billing_sync_loop_continues_after_sync_error():
    from app.services.billing_sync import billing_sync_loop

    call_count = 0

    async def _failing_sync(r, l, url):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise Exception("llogr unreachable")

    sleep_count = 0

    async def _fake_sleep(n):
        nonlocal sleep_count
        sleep_count += 1
        if sleep_count >= 3:
            raise asyncio.CancelledError

    with patch("app.services.billing_sync.sync_from_llogr", side_effect=_failing_sync), \
         patch("asyncio.sleep", side_effect=_fake_sleep):
        try:
            await billing_sync_loop(MagicMock(), _LIMITS, LLOGR_URL, interval=60)
        except asyncio.CancelledError:
            pass

    # Loop must survive the first sync error and attempt again
    assert call_count >= 2
