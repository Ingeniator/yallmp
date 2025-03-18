from fastapi import Request
from fastapi.responses import JSONResponse, Response
from httpx import AsyncClient, ConnectError, RequestError, Timeout, Limits
import random
import time
import asyncio
from app.core.config import settings
from app.core.logging_config import setup_logging

logger = setup_logging()

# Circuit breaker state (Use lock for thread safety)
circuit_open = False
circuit_open_time = 0
failure_timestamps = []
circuit_lock = asyncio.Lock()  # Prevent race conditions

cert = None
if settings.proxy_authorization_type == "CERT":
    cert = (settings.proxy_api_cert_path, settings.proxy_api_cert_path)

# Shared AsyncClient
client = AsyncClient(
    cert=cert,
    timeout=Timeout(5.0),  # default is 5.0 seconds)
    limits=Limits(
        max_connections=10,  # Maximum simultaneous connections
        max_keepalive_connections=5  # Keepalive connections
    )
)

async def exponential_backoff_retry(func, *args, **kwargs):
    """Performs a request with exponential backoff on retryable errors."""
    global circuit_open, circuit_open_time, failure_timestamps

    async with circuit_lock:
        # Check if the circuit breaker is open
        if circuit_open and time.time() - circuit_open_time < settings.proxy_recovery_time:
            return JSONResponse(content={"error": "Circuit breaker open. Try later."}, status_code=503)
    
    for attempt in range(settings.proxy_max_retries):
        try:
            response = await func(*args, **kwargs)

            # Handle 429 (Too Many Requests) with Retry-After
            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After")
                delay = int(retry_after) if retry_after else settings.proxy_base_delay * (settings.proxy_backoff_factor ** attempt) + random.uniform(0, 0.1)
                logger.warning(f"Rate limited. Retrying in {delay:.2f} seconds.")
                await asyncio.sleep(delay)
                continue  # Retry again
            
            if response.status_code not in {429, 500, 502, 503, 504}:
                async with circuit_lock:
                    failure_timestamps.clear()  # Reset failure timestamps on success
                return response  # Return successful response
            
        except ConnectError as e:
            # Connection failure details
            error_response = {
                "error": {
                    "message": "Connection failed",
                    "details": {
                        "exception": str(e),
                        "exception_type": type(e).__name__,
                        "attempt": attempt + 1
                    }
                }
            }
            logger.warning("Connection failed", details=error_response)
        except RequestError as e:
            # General request failure details
            error_response = {
                "error": {
                    "message": "Request failed",
                    "details": {
                        "exception": str(e),
                        "exception_type": type(e).__name__,
                        "attempt": attempt + 1
                    }
                }
            }
            logger.warning("Request failed", details=error_response)
        
        # Track failures
        current_time = time.time()
        failure_timestamps.append(current_time)
        window_start = current_time - settings.proxy_window_size
        failure_timestamps = [ts for ts in failure_timestamps if ts > window_start]

        async with circuit_lock:
            if len(failure_timestamps) >= settings.proxy_failure_threshold:
                circuit_open = True
                circuit_open_time = time.time()
                logger.error("Circuit breaker activated due to multiple failures.", details=error_response)
                return JSONResponse(content={"error": "Circuit breaker activated. Try later."}, status_code=503)

        # Exponential backoff delay
        delay = settings.proxy_base_delay * (settings.proxy_backoff_factor ** attempt) + random.uniform(0, 0.1)
        logger.debug(f"Retrying request in {delay:.2f} seconds.")
        await asyncio.sleep(delay)

    logger.error("Max retries exceeded.")
    return JSONResponse(content={"error": "Max retries exceeded"}, status_code=500)

async def proxy_request_with_retries(path: str, request: Request, custom_headers: dict[str, str] = {}):
    target_url = f"{settings.proxy_target_url}/{path}"
    method = request.method
    headers = dict(request.headers)
    headers.update(custom_headers)  # Inject additional headers
    body = await request.body()

    try:
        response = await exponential_backoff_retry(
            client.request, method, target_url, headers=headers, content=body
        )

        if response.status_code in {200, 201, 202, 203, 204, 205, 206, 207, 208, 226}:
            logger.info(f"Proxy request successful: {method} {target_url} -> {response.status_code}")

            # Create a new response with the correct headers
            response_headers = {key: value for key, value in response.headers.items()}
            response_headers.update(custom_headers)  # Inject custom headers

            return Response(
                content=response.content,
                status_code=response.status_code,
                headers=response_headers
            )

        else:
            error_response = {
                "error": {
                    "status_code": response.status_code,
                    "message": "Proxy request failed",
                    "details": {
                        "target_url": target_url,
                        "method": method,
                        "circuit_breaker_state": "open" if response.status_code == 503 else "unknown",
                    }
                }
            }
            logger.error("Proxy error details", details=error_response)
            return JSONResponse(content=error_response, status_code=response.status_code)

    except Exception as e:
        error_response = {
            "error": {
                "status_code": 500,
                "message": "Proxy request failed",
                "details": {
                    "target_url": target_url,
                    "method": method,
                    "exception": str(e),
                    "exception_type": type(e).__name__
                }
            }
        }
        logger.error("Proxy exception", details=error_response)
        return JSONResponse(content=error_response, status_code=500)
