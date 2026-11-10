"""Billing: tier-based spend limits enforced via Redis counters."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import yaml

from app.core.logging_config import setup_logging
from app.core.security import sanitize_group_id

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

# ── Currencies ────────────────────────────────────────────────────────────────
#
# USD is the default currency: its counters keep the original key names
# (billing:group:{org}:{period}) and its limits are the tier's top-level
# group_limit / user_limit. Every other currency (e.g. RUB for GigaChat) is
# counted separately under the same key with a ":{CURRENCY}" suffix and limited
# by the tier's optional ``currency_limits`` entry:
#
#   tier1:
#     group_limit: 100.0        # USD
#     user_limit: 10.0          # USD
#     currency_limits:
#       RUB: {group_limit: 10000.0, user_limit: 1000.0}
#
# A currency without a currency_limits entry is counted but not limited.

DEFAULT_CURRENCY = "USD"


def normalize_currency(currency: str | None) -> str:
    return (currency or DEFAULT_CURRENCY).strip().upper() or DEFAULT_CURRENCY


def _suffix(currency: str | None) -> str:
    cur = normalize_currency(currency)
    return "" if cur == DEFAULT_CURRENCY else f":{cur}"


def group_key(org: str, pk: str, currency: str | None = None) -> str:
    return f"billing:group:{org}:{pk}{_suffix(currency)}"


def user_key(group_id: str, pk: str, currency: str | None = None) -> str:
    return f"billing:user:{group_id}:{pk}{_suffix(currency)}"


def currency_limits(tier: dict, currency: str | None) -> dict | None:
    """Limits for *currency* within *tier*, or None when it is not limited."""
    cur = normalize_currency(currency)
    if cur == DEFAULT_CURRENCY:
        return tier
    extra = (tier.get("currency_limits") or {}).get(cur)
    if not extra:
        return None
    return {
        "group_limit": float(extra.get("group_limit", float("inf"))),
        "user_limit": float(extra.get("user_limit", float("inf"))),
        "alert_threshold": extra.get("alert_threshold", tier.get("alert_threshold", 0.8)),
    }


def limited_currencies(tier: dict) -> list[str]:
    """USD first, then every currency with configured limits."""
    others = sorted(
        normalize_currency(c) for c in (tier.get("currency_limits") or {})
    )
    return [DEFAULT_CURRENCY] + [c for c in others if c != DEFAULT_CURRENCY]

# ── Hot-path helpers ──────────────────────────────────────────────────────────

async def charge(
    redis,
    limits: dict,
    group_id: str,
    cost: float,
    currency: str | None = DEFAULT_CURRENCY,
) -> str:
    """
    Increment Redis counters for group and user after a successful request.
    Costs are counted per currency (see "Currencies" above); USD is the default.
    Returns a warning string if approaching limit, else empty string.

    ``group_id`` is sanitized the same way BillingMiddleware sanitizes it, so the
    keys charged here are the keys the pre-request limit check reads.
    """
    if not cost or cost <= 0:
        return ""
    group_id = sanitize_group_id(group_id)
    org = group_id.split("/")[0]
    tier = get_tier(limits, org)
    pk = period_key(tier["period"])
    ttl = period_ttl(tier["period"])

    try:
        gkey = group_key(org, pk, currency)
        group_total = float(await redis.incrbyfloat(gkey, cost))
        await redis.expire(gkey, ttl, xx=False)  # set TTL only if key is new

        if "/" in group_id:
            ukey = user_key(group_id, pk, currency)
            await redis.incrbyfloat(ukey, cost)
            await redis.expire(ukey, ttl, xx=False)

        lim = currency_limits(tier, currency)
        if lim and group_total >= lim["group_limit"] * lim.get("alert_threshold", 0.8):
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
            keys = [k async for k in redis.scan_iter(f"billing:group:*:{pk}")]
        elif is_org_admin:
            keys = [k async for k in redis.scan_iter(f"billing:group:{org}*:{pk}")]
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
                "period": pk,
                "group_limit": key_tier["group_limit"],
                "group_spent": round(spent, 6),
                "group_pct": round(spent / key_tier["group_limit"] * 100, 1) if key_tier["group_limit"] else 0,
                "alert": spent >= key_tier["group_limit"] * key_tier.get("alert_threshold", 0.8),
            })

        # Per-user breakdown for admins
        users = []
        if is_org_admin or is_super_admin:
            user_pattern = f"billing:user:{org}/*:{pk}" if is_org_admin else f"billing:user:*:{pk}"
            async for ukey in redis.scan_iter(user_pattern):
                uval = await redis.get(ukey)
                u_spent = float(uval or 0)
                parts = ukey.split(":")
                u_group_id = parts[2] if len(parts) >= 4 else ""
                u_org = u_group_id.split("/")[0]
                u_tier = get_tier(limits, u_org)
                users.append({
                    "group_id": u_group_id,
                    "user_limit": u_tier["user_limit"],
                    "user_spent": round(u_spent, 6),
                    "user_pct": round(u_spent / u_tier["user_limit"] * 100, 1) if u_tier["user_limit"] else 0,
                })
            users.sort(key=lambda x: x["user_spent"], reverse=True)

        # Current user's own spend
        user_spent = 0.0
        user_limit = tier["user_limit"]
        if "/" in (group_id or ""):
            user_val = await redis.get(f"billing:user:{group_id}:{pk}")
            user_spent = float(user_val or 0)

        return {
            "period": pk,
            "groups": groups,
            "users": users,
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
