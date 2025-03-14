import httpx
from app.core.logging_config import setup_logging
import asyncio
import random
import time
from fastapi import FastAPI, Request
from starlette.responses import Response, JSONResponse
from collections import deque

logger = setup_logging()

app = FastAPI()

# Target backend server to forward requests
TARGET_URL = "http://localhost:8000/fake"

# Custom headers to add
CUSTOM_HEADERS = {
    "X-Custom-Header": "MyCustomValue"
}

# Retry configuration
MAX_RETRIES = 5
BASE_DELAY = 0.5  # Base delay in seconds
BACKOFF_FACTOR = 2  # Exponential backoff multiplier

# Circuit breaker configuration
FAILURE_THRESHOLD = 5  # Number of failures before tripping
RECOVERY_TIME = 30  # Cooldown period (seconds)
WINDOW_SIZE = 60  # Sliding window size in seconds

# Track failures in a sliding window
failure_timestamps = deque()

# Circuit breaker state
circuit_open = False
circuit_open_time = 0

async def exponential_backoff_retry(request_func, *args, **kwargs):
    """Performs a request with exponential backoff on retryable errors."""
    global circuit_open, circuit_open_time

    # Check if the circuit breaker is open
    if circuit_open and time.time() - circuit_open_time < RECOVERY_TIME:
        return JSONResponse({"error": "Circuit breaker open. Try later."}, status_code=503)
    
    for attempt in range(MAX_RETRIES):
        try:
            response = await request_func(*args, **kwargs)

            # Handle 429 with Retry-After
            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After")
                delay = int(retry_after) if retry_after else BASE_DELAY * (BACKOFF_FACTOR ** attempt) + random.uniform(0, 0.1)
                await asyncio.sleep(delay)
                continue  # Retry again
            
            if response.status_code not in {429, 500, 502, 503, 504}:
                return response  # Return response if it's not a failure
            
        except httpx.RequestError:
            pass  # Consider it a failure

        # Log failure timestamp
        failure_timestamps.append(time.time())

        # Remove old failures outside of the window
        while failure_timestamps and failure_timestamps[0] < time.time() - WINDOW_SIZE:
            failure_timestamps.popleft()

        # Check if failure threshold is exceeded
        if len(failure_timestamps) >= FAILURE_THRESHOLD:
            circuit_open = True
            circuit_open_time = time.time()
            return JSONResponse({"error": "Circuit breaker activated. Try later."}, status_code=503)

        # Exponential backoff delay
        delay = BASE_DELAY * (BACKOFF_FACTOR ** attempt) + random.uniform(0, 0.1)
        await asyncio.sleep(delay)

    return JSONResponse({"error": "Max retries exceeded"}, status_code=500)

async def proxy_request_with_retries(path: str, request: Request):
    client = httpx.AsyncClient()
    
    target_url = f"{TARGET_URL}/{path}"
    method = request.method
    headers = dict(request.headers)
    headers.update(CUSTOM_HEADERS)  # Inject additional headers
    body = await request.body()
    
    try:
        # Perform request with retries
        response = await exponential_backoff_retry(
            client.request, method, target_url, headers=headers, content=body
        )

        # Inject custom response headers
        for key, value in CUSTOM_HEADERS.items():
            response.headers[key] = value

        return Response(content=response.content, status_code=response.status_code, headers=dict(response.headers))
    except Exception as e:
        logger.error(f"Proxy error: {e}")
        return JSONResponse({"error": "Proxy request failed"}, status_code=500)
