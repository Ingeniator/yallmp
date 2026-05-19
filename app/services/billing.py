"""Billing: tier-based spend limits enforced via Redis counters."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import yaml

from app.core.logging_config import setup_logging

logger = setup_logging()

# ── Config ────────────────────────────────────────────────────────────────────

def load_limits(path: str) -> dict:
    """Load limits.yaml once on startup."""
    try:
        return yaml.safe_load(Path(path).read_text())
    except Exception as exc:
        logger.warning("billing_limits_load_failed", path=path, error=str(exc))
        return {"tiers": {}, "orgs": {}}


def get_tier(limits: dict, org: str) -> dict:
    """Resolve tier config for org, fallback to 'default'."""
    orgs = limits.get("orgs", {})
    tiers = limits.get("tiers", {})
    tier_name = orgs.get(org) or orgs.get("default", "")
    return tiers.get(tier_name, {
        "period": "month",
        "group_limit": 999999.0,
        "user_limit": 999999.0,
        "alert_threshold": 1.0,
    })


def period_key(period: str) -> str:
    """Return Redis-safe period key: 'month' → '2026-05', 'week' → '2026-W21'."""
    now = datetime.now(timezone.utc)
    if period == "week":
        return now.strftime("%Y-W%W")
    return now.strftime("%Y-%m")


def period_ttl(period: str) -> int:
    """Return seconds until end of current period (for Redis key TTL)."""
    from calendar import monthrange
    now = datetime.now(timezone.utc)
    if period == "week":
        days_left = 6 - now.weekday()
        return days_left * 86400 + (86400 - now.hour * 3600 - now.minute * 60 - now.second)
    _, last_day = monthrange(now.year, now.month)
    end = now.replace(day=last_day, hour=23, minute=59, second=59)
    return max(int((end - now).total_seconds()), 60)

# ── Hot-path helpers ──────────────────────────────────────────────────────────

async def charge(redis, limits: dict, group_id: str, cost: float) -> str:
    """
    Increment Redis counters for group and user after a successful request.
    Returns a warning string if approaching limit, else empty string.
    """
    if not cost or cost <= 0:
        return ""
    org = group_id.split("/")[0]
    tier = get_tier(limits, org)
    pk = period_key(tier["period"])
    ttl = period_ttl(tier["period"])

    try:
        group_key = f"billing:group:{org}:{pk}"
        group_total = float(await redis.incrbyfloat(group_key, cost))
        await redis.expire(group_key, ttl, xx=False)  # set TTL only if key is new

        if "/" in group_id:
            user_key = f"billing:user:{group_id}:{pk}"
            await redis.incrbyfloat(user_key, cost)
            await redis.expire(user_key, ttl, xx=False)

        threshold = tier.get("alert_threshold", 0.8)
        if group_total >= tier["group_limit"] * threshold:
            return "approaching group limit"
    except Exception as exc:
        logger.warning("billing_charge_failed", group_id=group_id, error=str(exc))
    return ""


async def get_billing_summary(redis, limits: dict, group_id: str, role: str) -> dict:
    """Return current period spend for dashboard API."""
    is_super_admin = role == "SUPER_ADMIN"
    is_org_admin = role == "ORG_ADMIN"

    org = group_id.split("/")[0] if group_id else "unknown"
    tier = get_tier(limits, org)
    pk = period_key(tier["period"])

    try:
        if is_super_admin:
            # Scan all group keys for this period
            keys = [k.decode() async for k in redis.scan_iter(f"billing:group:*:{pk}")]
        elif is_org_admin:
            keys = [k.decode() async for k in redis.scan_iter(f"billing:group:{org}*:{pk}")]
        else:
            keys = [f"billing:group:{org}:{pk}"]

        groups = []
        for key in keys:
            val = await redis.get(key)
            spent = float(val or 0)
            key_org = key.split(":")[2] if key.count(":") >= 2 else org
            key_tier = get_tier(limits, key_org)
            groups.append({
                "org": key_org,
                "tier": next(
                    (k for k, v in limits.get("orgs", {}).items() if v == next(
                        (t for t, cfg in limits.get("tiers", {}).items() if cfg == key_tier), ""
                    )), ""),
                "period": pk,
                "group_limit": key_tier["group_limit"],
                "group_spent": round(spent, 6),
                "group_pct": round(spent / key_tier["group_limit"] * 100, 1) if key_tier["group_limit"] else 0,
                "alert": spent >= key_tier["group_limit"] * key_tier.get("alert_threshold", 0.8),
            })

        # User spend for own group_id
        user_spent = 0.0
        user_limit = tier["user_limit"]
        if "/" in (group_id or ""):
            user_val = await redis.get(f"billing:user:{group_id}:{pk}")
            user_spent = float(user_val or 0)

        return {
            "period": pk,
            "groups": groups,
            "user": {
                "group_id": group_id,
                "user_limit": user_limit,
                "user_spent": round(user_spent, 6),
                "user_pct": round(user_spent / user_limit * 100, 1) if user_limit else 0,
            } if "/" in (group_id or "") else None,
        }
    except Exception as exc:
        logger.warning("billing_summary_failed", error=str(exc))
        return {"period": pk, "groups": [], "user": None}
