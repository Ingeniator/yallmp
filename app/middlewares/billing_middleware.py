"""Billing middleware: pre-request spend limit enforcement."""
from __future__ import annotations

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app.core.config import settings
from app.core.logging_config import setup_logging
from app.core.security import sanitize_group_id
from app.services.billing import get_tier, period_key

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

        try:
            group_key = f"billing:group:{org}:{pk}"
            current_group = float(await redis.get(group_key) or 0)
            if current_group >= tier["group_limit"]:
                logger.info("billing_group_limit_reached", group_id=group_id, spent=current_group)
                return JSONResponse(
                    status_code=429,
                    content={"error": "group spend limit reached", "spent": current_group, "limit": tier["group_limit"]},
                )

            if "/" in group_id:
                user_key = f"billing:user:{group_id}:{pk}"
                current_user = float(await redis.get(user_key) or 0)
                if current_user >= tier["user_limit"]:
                    logger.info("billing_user_limit_reached", group_id=group_id, spent=current_user)
                    return JSONResponse(
                        status_code=429,
                        content={"error": "user spend limit reached", "spent": current_user, "limit": tier["user_limit"]},
                    )
        except Exception as exc:
            logger.warning("billing_check_failed", group_id=group_id, error=str(exc))
            # Fail open — LLM access is more critical than a missed limit check

        response = await call_next(request)

        # Soft warning header (based on pre-request read — no extra Redis call)
        try:
            current_group = float(await redis.get(f"billing:group:{org}:{pk}") or 0)
            if current_group >= tier["group_limit"] * tier.get("alert_threshold", 0.8):
                response.headers["X-Billing-Warning"] = "approaching group limit"
        except Exception:
            pass

        return response
