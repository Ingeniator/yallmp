from app.core.logging_config import setup_logging
from app.core.security import redact_headers
import logging
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
import time
import json

logger = setup_logging()


class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        """Middleware to log detailed request and response information"""

        # skip logging stream requests
        content_type = request.headers.get("content-type", "").lower()
        transfer_encoding = request.headers.get("transfer-encoding", "").lower()
        is_stream = (request.method == "POST" and (content_type.startswith("multipart/form-data") or "chunked" in transfer_encoding))
        if is_stream:
            logger.info("Middleware has skipped stream requests")
            return await call_next(request)

        start_time = time.time()

        headers = redact_headers(dict(request.headers))

        log_data = {
            "method": request.method,
            "path": request.url.path,
            "query_params": dict(request.query_params),
            "headers": headers,
        }

        if logger.isEnabledFor(logging.DEBUG):
            try:
                body = await request.body()
                body_str = body.decode("utf-8", errors="replace") if body else None
                log_data["body"] = body_str
            except Exception as e:
                log_data["body_error"] = f"Cannot log body: {e}"

        logger.debug("Incoming Request", **log_data)

        response = await call_next(request)

        process_time = time.time() - start_time

        if response.media_type == "text/event-stream":
            logger.info("Streaming response", status_code=response.status_code, process_time=f"{process_time:.4f}s")
        elif logger.isEnabledFor(logging.DEBUG):
            # Only consume response body for DEBUG logging
            response_body = b""
            async for chunk in response.body_iterator:
                response_body += chunk

            response = Response(
                content=response_body,
                status_code=response.status_code,
                headers=dict(response.headers),
                media_type=response.media_type
            )

            response_log = {
                "status_code": response.status_code,
                "headers": dict(response.headers),
                "process_time": f"{process_time:.4f}s",
            }
            try:
                response_log["body"] = json.loads(response_body.decode("utf-8"))
            except (ValueError, UnicodeDecodeError):
                response_log["body"] = response_body.decode("utf-8", errors="replace")

            logger.debug("Outgoing Response", **response_log)
        else:
            logger.info("Request completed", status_code=response.status_code, process_time=f"{process_time:.4f}s")

        return response
