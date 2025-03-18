import httpx
from app.core.logging_config import setup_logging
import asyncio
import random
import time
from fastapi import Request
from starlette.responses import Response, JSONResponse
from collections import deque
from app.core.config import settings
logger = setup_logging()

# Track failures in a sliding window
failure_timestamps = deque()

# Circuit breaker state
circuit_open = False
circuit_open_time = 0

async def exponential_backoff_retry(request_func, *args, **kwargs):
    """Performs a request with exponential backoff on retryable errors."""
    global circuit_open, circuit_open_time

    # Check if the circuit breaker is open
    if circuit_open and time.time() - circuit_open_time < settings.proxy_recovery_time:
        return JSONResponse({"error": "Circuit breaker open. Try later."}, status_code=503)
    
    for attempt in range(settings.proxy_max_retries):
        try:
            response = await request_func(*args, **kwargs)

            # Handle 429 with Retry-After
            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After")
                delay = int(retry_after) if retry_after else settings.proxy_base_delay * (settings.proxy_backoff_factor ** attempt) + random.uniform(0, 0.1)
                await asyncio.sleep(delay)
                continue  # Retry again
            
            if response.status_code not in {429, 500, 502, 503, 504}:
                return response  # Return response if it's not a failure
            
        except httpx.RequestError:
            pass  # Consider it a failure

        # Log failure timestamp
        failure_timestamps.append(time.time())

        # Remove old failures outside of the window
        while failure_timestamps and failure_timestamps[0] < time.time() - settings.proxy_window_size:
            failure_timestamps.popleft()

        # Check if failure threshold is exceeded
        if len(failure_timestamps) >= settings.proxy_failure_threshold:
            circuit_open = True
            circuit_open_time = time.time()
            return JSONResponse({"error": "Circuit breaker activated. Try later."}, status_code=503)

        # Exponential backoff delay
        delay = settings.proxy_base_delay * (settings.proxy_backoff_factor ** attempt) + random.uniform(0, 0.1)
        await asyncio.sleep(delay)

    return JSONResponse({"error": "Max retries exceeded"}, status_code=500)

async def proxy_request_with_retries(path: str, request: Request, custom_headers: dict[str, str] = {}):
    if settings.proxy_authorization_type == "CERT":
        cert = (settings.proxy_api_cert_path, settings.proxy_api_cert_path)
    else:
        cert = None
    client = httpx.AsyncClient(cert=cert)
    
    target_url = f"{settings.proxy_target_url}/{path}"
    method = request.method
    headers = dict(request.headers)
    headers.update(custom_headers)  # Inject additional headers
    body = await request.body()
    
    try:
        # Perform request with retries
        response = await exponential_backoff_retry(
            client.request, method, target_url, headers=headers, content=body
        )

        # Inject custom response headers
        for key, value in custom_headers.items():
            response.headers[key] = value

        return Response(content=response.content, status_code=response.status_code, headers=dict(response.headers))
    except Exception as e:
        logger.error(f"Proxy error: {e}")
        return JSONResponse({"error": "Proxy request failed"}, status_code=500)
