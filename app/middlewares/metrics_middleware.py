import re
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST, multiprocess, CollectorRegistry
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
import time
from app.core.logging_config import setup_logging

# Normalize dynamic path segments (UUIDs, numeric IDs) to prevent unbounded label cardinality
_PATH_ID_RE = re.compile(r'/[0-9a-f]{8,}(?:-[0-9a-f]{4,}){0,4}|/\d+')

# Request count metric
REQUEST_COUNT = Counter(
    "http_requests_total", "Total number of HTTP requests",
    ["method", "endpoint", "status_code"]
)

# Request duration metric
REQUEST_DURATION = Histogram(
    "http_request_duration_seconds", "Histogram of request processing time",
    ["method", "endpoint"]
)

logger = setup_logging()


def _normalize_path(path: str) -> str:
    """Replace dynamic path segments with placeholders to limit cardinality."""
    return _PATH_ID_RE.sub("/:id", path)


class PrometheusMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):

        # skip metrics for stream requests
        content_type = request.headers.get("content-type", "").lower()
        transfer_encoding = request.headers.get("transfer-encoding", "").lower()
        is_stream = (request.method == "POST" and (content_type.startswith("multipart/form-data") or "chunked" in transfer_encoding))
        if is_stream:
            logger.info("Middleware has skipped stream requests")
            return await call_next(request)

        method = request.method
        endpoint = _normalize_path(request.url.path)

        start_time = time.time()
        response = await call_next(request)
        duration = time.time() - start_time

        REQUEST_COUNT.labels(method=method, endpoint=endpoint, status_code=response.status_code).inc()
        REQUEST_DURATION.labels(method=method, endpoint=endpoint).observe(duration)

        return response

registry = CollectorRegistry()
multiprocess.MultiProcessCollector(registry)


async def metrics():
    return Response(content=generate_latest(registry), media_type=CONTENT_TYPE_LATEST)
