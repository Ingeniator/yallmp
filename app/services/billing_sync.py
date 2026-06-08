"""Seed Redis billing counters from llogr's ClickHouse-backed billing API.

Runs once on startup and then on a periodic loop. Each sync overwrites Redis
keys using a Lua script that only raises the value, never lowers it, so Redis
stays consistent with ClickHouse while in-flight increments from concurrent
requests are never silently discarded.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import httpx

from app.core.logging_config import setup_logging
from app.services.billing import get_tier, period_key, period_ttl

logger = setup_logging()

_sync_lock = asyncio.Lock()
_PAGE_SIZE = 500

# Atomically sets key only if the new value is greater than the current value.
_LUA_SET_GT = """
local c = redis.call('GET', KEYS[1])
if not c or tonumber(ARGV[1]) > tonumber(c) then
    return redis.call('SET', KEYS[1], ARGV[1], 'EX', tonumber(ARGV[2]))
end
return 0
"""


def _period_range(period_type: str, now: datetime) -> tuple[str, str]:
    """Return (start, end) ISO strings for the current period window."""
    if period_type == "week":
        start = (now - timedelta(days=now.weekday())).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        end = start + timedelta(days=7)
    else:  # month
        start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        if now.month == 12:
            end = now.replace(year=now.year + 1, month=1, day=1,
                              hour=0, minute=0, second=0, microsecond=0)
        else:
            end = now.replace(month=now.month + 1, day=1,
                              hour=0, minute=0, second=0, microsecond=0)
    fmt = "%Y-%m-%dT%H:%M:%S"
    return start.strftime(fmt), end.strftime(fmt)


async def _fetch_page(llogr_url: str, start: str, end: str, offset: int) -> dict:
    url = f"{llogr_url.rstrip('/')}/api/public/billing/summary"
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(
            url,
            params={"start": start, "end": end, "user_limit": _PAGE_SIZE, "user_offset": offset},
            headers={"X-Group-ID": "system", "X-Role": "SUPER_ADMIN"},
        )
        resp.raise_for_status()
        return resp.json()


async def sync_from_llogr(redis, limits: dict, llogr_url: str) -> None:
    """Overwrite Redis billing counters with authoritative values from llogr.

    Uses SET ... GT so the sync can only raise a counter, never lower it.
    This means a burst of in-flight requests right before the sync runs will
    not be silently reversed by a slightly-lagging ClickHouse value.
    """
    if _sync_lock.locked():
        logger.info("billing_sync_skipped", reason="previous_sync_still_running")
        return

    async with _sync_lock:
        now = datetime.now(timezone.utc)

        # Collect distinct period types used across configured tiers
        period_types: set[str] = {
            t.get("period", "month") for t in limits.get("tiers", {}).values()
        } or {"month"}

        total_groups = total_users = 0

        for pt in period_types:
            start, end = _period_range(pt, now)
            pipe = redis.pipeline()
            groups_written = False

            offset = 0
            while True:
                try:
                    page = await _fetch_page(llogr_url, start, end, offset)
                except Exception as exc:
                    logger.warning("billing_sync_fetch_failed", period=pt, offset=offset,
                                   url=llogr_url, error=str(exc))
                    break

                # Groups are returned in full on every page — only write on first fetch
                if not groups_written:
                    for g in page.get("groups", []):
                        org = g.get("org", "")
                        spent = float(g.get("group_spent") or 0)
                        if not org:
                            continue
                        tier = get_tier(limits, org)
                        pk = period_key(tier["period"])
                        pipe.eval(_LUA_SET_GT, 1, f"billing:group:{org}:{pk}",
                                  spent, period_ttl(tier["period"]))
                        total_groups += 1
                    groups_written = True

                users = page.get("users", [])
                for u in users:
                    project_id = u.get("project_id", "")
                    spent = float(u.get("user_spent") or 0)
                    if not project_id or "/" not in project_id:
                        continue
                    org = project_id.split("/")[0]
                    tier = get_tier(limits, org)
                    pk = period_key(tier["period"])
                    pipe.eval(_LUA_SET_GT, 1, f"billing:user:{project_id}:{pk}",
                              spent, period_ttl(tier["period"]))
                    total_users += 1

                if not page.get("has_more", False):
                    break
                offset += _PAGE_SIZE

            try:
                await pipe.execute()
            except Exception as exc:
                logger.warning("billing_sync_redis_write_failed", period=pt, error=str(exc))

        logger.info("billing_sync_complete", groups=total_groups, users=total_users)


async def billing_sync_loop(redis, limits: dict, llogr_url: str, interval: int) -> None:
    """Periodic background task: sync every `interval` seconds."""
    while True:
        await asyncio.sleep(interval)
        try:
            await sync_from_llogr(redis, limits, llogr_url)
        except Exception as exc:
            logger.warning("billing_sync_loop_unhandled_error", error=str(exc))
