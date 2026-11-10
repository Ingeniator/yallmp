"""Billing middleware: pre-request spend limit enforcement."""
from __future__ import annotations

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app.core.config import settings
from app.core.logging_config import setup_logging
from app.core.security import sanitize_group_id
from app.services.billing import (
    DEFAULT_CURRENCY,
    currency_limits,
    get_tier,
    group_key,
    limited_currencies,
    period_key,
    user_key,
)

logger = setup_logging()

_LLM_PATH_PREFIX = f"{settings.root_path}/llm/"


class BillingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if not settings.billing_enabled:
            return await call_next(request)

        if not request.url.path.startswith(_LLM_PATH_PREFIX):
            return await call_next(request)

        redis = getattr(request.app.state, "billing_redis", None)
        limits = getattr(request.app.state, "billing_limits", {})
        if not redis or not limits:
            return await call_next(request)

        group_id = sanitize_group_id(request.headers.get("x-group-id"))
        org = group_id.split("/")[0]
        tier = get_tier(limits, org)
        pk = period_key(tier["period"])

        # Limits are per currency (USD default, e.g. RUB separately). The
        # provider — and so the currency — is not known before routing, so an
        # exhausted budget in any limited currency blocks the request.
        currencies = limited_currencies(tier)
        try:
            for cur in currencies:
                lim = currency_limits(tier, cur)
                extra = {} if cur == DEFAULT_CURRENCY else {"currency": cur}

                current_group = float(await redis.get(group_key(org, pk, cur)) or 0)
                if current_group >= lim["group_limit"]:
                    logger.info("billing_group_limit_reached", group_id=group_id,
                                spent=current_group, currency=cur)
                    return JSONResponse(
                        status_code=429,
                        content={"error": "group spend limit reached", "spent": current_group,
                                 "limit": lim["group_limit"], **extra},
                    )

                if "/" in group_id:
                    current_user = float(await redis.get(user_key(group_id, pk, cur)) or 0)
                    if current_user >= lim["user_limit"]:
                        logger.info("billing_user_limit_reached", group_id=group_id,
                                    spent=current_user, currency=cur)
                        return JSONResponse(
                            status_code=429,
                            content={"error": "user spend limit reached", "spent": current_user,
                                     "limit": lim["user_limit"], **extra},
                        )
        except Exception as exc:
            logger.warning("billing_check_failed", group_id=group_id, error=str(exc))
            # Fail open — LLM access is more critical than a missed limit check

        response = await call_next(request)

        # Soft warning header
        try:
            for cur in currencies:
                lim = currency_limits(tier, cur)
                current_group = float(await redis.get(group_key(org, pk, cur)) or 0)
                if current_group >= lim["group_limit"] * lim.get("alert_threshold", 0.8):
                    response.headers["X-Billing-Warning"] = (
                        "approaching group limit" if cur == DEFAULT_CURRENCY
                        else f"approaching group limit ({cur})"
                    )
                    break
        except Exception:
            pass

        return response
