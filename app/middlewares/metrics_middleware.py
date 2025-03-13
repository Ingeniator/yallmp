from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
import time

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

# Middleware for collecting metrics
class PrometheusMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        method = request.method
        endpoint = request.url.path

        start_time = time.time()
        response = await call_next(request)
        duration = time.time() - start_time

        REQUEST_COUNT.labels(method=method, endpoint=endpoint, status_code=response.status_code).inc()
        REQUEST_DURATION.labels(method=method, endpoint=endpoint).observe(duration)

        return response

# Metrics endpoint handler
async def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

