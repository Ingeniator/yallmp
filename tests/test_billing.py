"""Unit tests for yallmp billing service."""
from __future__ import annotations

import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch


# ── load_limits ───────────────────────────────────────────────────────────────

def test_load_limits_returns_parsed_yaml(tmp_path: Path):
    from app.services.billing import load_limits

    f = tmp_path / "limits.yaml"
    f.write_text(
        "tiers:\n"
        "  tier1:\n"
        "    period: month\n"
        "    group_limit: 100.0\n"
        "    user_limit: 10.0\n"
        "    alert_threshold: 0.8\n"
        "orgs:\n"
        "  myorg: tier1\n"
    )
    limits = load_limits(str(f))
    assert limits["tiers"]["tier1"]["group_limit"] == 100.0
    assert limits["orgs"]["myorg"] == "tier1"


def test_load_limits_missing_file_returns_empty():
    from app.services.billing import load_limits

    limits = load_limits("/nonexistent/limits.yaml")
    assert limits == {"tiers": {}, "orgs": {}}


# ── get_tier ──────────────────────────────────────────────────────────────────

_LIMITS = {
    "tiers": {
        "tier1": {"period": "month", "group_limit": 100.0, "user_limit": 10.0, "alert_threshold": 0.8},
        "tier_big": {"period": "month", "group_limit": 1000.0, "user_limit": 50.0, "alert_threshold": 0.9},
    },
    "orgs": {
        "acme": "tier_big",
        "default": "tier1",
    },
}


def test_get_tier_known_org():
    from app.services.billing import get_tier

    tier = get_tier(_LIMITS, "acme")
    assert tier["group_limit"] == 1000.0


def test_get_tier_falls_back_to_default():
    from app.services.billing import get_tier

    tier = get_tier(_LIMITS, "unknown-org")
    assert tier["group_limit"] == 100.0


def test_get_tier_no_limits_returns_unlimited():
    from app.services.billing import get_tier

    tier = get_tier({}, "any-org")
    assert tier["group_limit"] == 999999.0
    assert tier["user_limit"] == 999999.0


# ── period_key ────────────────────────────────────────────────────────────────

def test_period_key_month(freezegun_patch):
    from app.services.billing import period_key

    with patch("app.services.billing.datetime") as mock_dt:
        from datetime import datetime, timezone
        mock_dt.now.return_value = datetime(2026, 5, 19, tzinfo=timezone.utc)
        mock_dt.now.side_effect = None
        assert period_key("month") == "2026-05"


def test_period_key_week(freezegun_patch):
    from app.services.billing import period_key

    with patch("app.services.billing.datetime") as mock_dt:
        from datetime import datetime, timezone
        mock_dt.now.return_value = datetime(2026, 5, 19, tzinfo=timezone.utc)
        mock_dt.now.side_effect = None
        key = period_key("week")
        assert key.startswith("2026-W")


# ── charge ────────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_redis():
    r = AsyncMock()
    r.incrbyfloat = AsyncMock(return_value=5.0)
    r.expire = AsyncMock()
    r.get = AsyncMock(return_value=None)
    return r


@pytest.mark.asyncio
async def test_charge_increments_group_and_user(mock_redis):
    from app.services.billing import charge

    warning = await charge(mock_redis, _LIMITS, "acme/alice", 1.5)
    assert mock_redis.incrbyfloat.call_count == 2  # group + user
    calls = [str(c) for c in mock_redis.incrbyfloat.call_args_list]
    assert any("billing:group:acme" in c for c in calls)
    assert any("billing:user:acme/alice" in c for c in calls)


@pytest.mark.asyncio
async def test_charge_only_group_when_no_user(mock_redis):
    from app.services.billing import charge

    await charge(mock_redis, _LIMITS, "acme", 1.0)
    assert mock_redis.incrbyfloat.call_count == 1


@pytest.mark.asyncio
async def test_charge_zero_cost_skips_redis(mock_redis):
    from app.services.billing import charge

    await charge(mock_redis, _LIMITS, "acme/alice", 0.0)
    mock_redis.incrbyfloat.assert_not_called()


@pytest.mark.asyncio
async def test_charge_returns_warning_when_threshold_reached(mock_redis):
    from app.services.billing import charge

    # "unknown-org" falls back to "default" → tier1: group_limit=100, threshold=0.8 → warn at 80
    mock_redis.incrbyfloat = AsyncMock(return_value=85.0)
    warning = await charge(mock_redis, _LIMITS, "unknown-org/alice", 1.0)
    assert warning == "approaching group limit"


@pytest.mark.asyncio
async def test_charge_returns_empty_below_threshold(mock_redis):
    from app.services.billing import charge

    mock_redis.incrbyfloat = AsyncMock(return_value=50.0)
    warning = await charge(mock_redis, _LIMITS, "acme/alice", 1.0)
    assert warning == ""


@pytest.mark.asyncio
async def test_charge_fails_open_on_redis_error(mock_redis):
    from app.services.billing import charge

    mock_redis.incrbyfloat.side_effect = ConnectionError("redis down")
    # Should not raise
    warning = await charge(mock_redis, _LIMITS, "acme/alice", 1.0)
    assert warning == ""


# ── get_billing_summary ───────────────────────────────────────────────────────

@pytest.fixture
def mock_redis_summary():
    r = AsyncMock()

    async def _scan_iter(pattern):
        if "group" in pattern:
            for k in [b"billing:group:acme:2026-05"]:
                yield k
        else:
            return
        return

    r.scan_iter = _scan_iter
    r.get = AsyncMock(return_value=b"42.5")
    return r


@pytest.mark.asyncio
async def test_get_billing_summary_regular_user(mock_redis_summary):
    from app.services.billing import get_billing_summary

    result = await get_billing_summary(mock_redis_summary, _LIMITS, "acme/alice", "USER")
    assert result["period"].startswith("2026")
    assert len(result["groups"]) == 1
    assert result["groups"][0]["org"] == "acme"
    assert result["user"] is not None
    assert result["user"]["group_id"] == "acme/alice"


@pytest.mark.asyncio
async def test_get_billing_summary_no_slash_group_id(mock_redis_summary):
    from app.services.billing import get_billing_summary

    result = await get_billing_summary(mock_redis_summary, _LIMITS, "acme", "USER")
    assert result["user"] is None


@pytest.mark.asyncio
async def test_get_billing_summary_redis_error_returns_empty(mock_redis_summary):
    from app.services.billing import get_billing_summary

    mock_redis_summary.get.side_effect = ConnectionError("redis down")
    result = await get_billing_summary(mock_redis_summary, _LIMITS, "acme/alice", "USER")
    assert result["groups"] == []
    assert result["user"] is None


# Dummy fixture — period_key tests use direct mocking instead
@pytest.fixture
def freezegun_patch():
    pass
