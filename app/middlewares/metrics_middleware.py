import re
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST, multiprocess, CollectorRegistry, REGISTRY
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
import os
import time
from app.core.logging_config import setup_logging

# Normalize dynamic path segments (UUIDs, numeric IDs) to prevent unbounded label cardinality
_PATH_ID_RE = re.compile(r'/[0-9a-f]{8,}(?:-[0-9a-f]{4,}){0,4}|/\d+')

# Request count metric
REQUEST_COUNT = Counter(
    "http_requests_total", "Total number of HTTP requests",
    ["method", "endpoint", "status_code", "group_id"]
)

# Request duration metric
REQUEST_DURATION = Histogram(
    "http_request_duration_seconds", "Histogram of request processing time",
    ["method", "endpoint", "group_id"]
)

logger = setup_logging()


def _normalize_path(path: str) -> str:
    """Replace dynamic path segments with placeholders to limit cardinality."""
    return _PATH_ID_RE.sub("/:id", path)


def get_metrics_registry() -> CollectorRegistry:
    """Return the correct registry for generating /metrics output.

    In multi-worker mode (PROMETHEUS_MULTIPROC_DIR is set and exists) we build
    a fresh CollectorRegistry with a MultiProcessCollector so that metrics from
    all workers are merged.  In single-worker mode we fall back to the default
    REGISTRY which already holds the Counter/Histogram instances above.
    """
    multiproc_dir = os.environ.get("PROMETHEUS_MULTIPROC_DIR")
    if multiproc_dir and os.path.isdir(multiproc_dir):
        registry = CollectorRegistry()
        multiprocess.MultiProcessCollector(registry)
        return registry
    return REGISTRY


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
        group_id = request.headers.get("x-group-id", "")

        start_time = time.time()
        response = await call_next(request)
        duration = time.time() - start_time

        REQUEST_COUNT.labels(method=method, endpoint=endpoint, status_code=response.status_code, group_id=group_id).inc()
        REQUEST_DURATION.labels(method=method, endpoint=endpoint, group_id=group_id).observe(duration)

        return response


async def metrics():
    registry = get_metrics_registry()
    return Response(content=generate_latest(registry), media_type=CONTENT_TYPE_LATEST)
