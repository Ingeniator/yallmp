from fastapi import Request
from fastapi.responses import JSONResponse
from httpx import AsyncClient, ConnectError, RequestError, Timeout, Limits, Response as HTTPXResponse, AsyncByteStream
import random
import time
import asyncio
from app.core.config import settings
from app.core.logging_config import setup_logging
from app.services.metrics_callback_handler import MetricsCallbackHandler
from app.schemas.prompt import ChainMetadataForTracking, ChainType
import json
import fnmatch
from typing import AsyncIterator
from app.core.security import redact_headers

logger = setup_logging()

# Circuit breaker state (Use lock for thread safety)
circuit_open = False
circuit_open_time = 0
failure_timestamps = []
circuit_lock = asyncio.Lock()  # Prevent race conditions

async def create_async_client():
    cert = None
    if settings.proxy_authorization_type == "CERT":
        cert = (settings.proxy_api_cert_path, settings.proxy_api_cert_key_path)
    # Shared AsyncClient
    return AsyncClient(
        cert=cert,
        timeout=Timeout(connect=settings.proxy_connect_timeout, read=settings.proxy_read_timeout, write=settings.proxy_write_timeout, pool=settings.proxy_pool_timeout),
        limits=Limits(
            max_connections=settings.max_connections,  # Maximum simultaneous connections
            max_keepalive_connections=settings.max_keepalive_connections  # Keepalive connections
        ),
        verify=settings.proxy_verify_ssl
    )

async def get_circuit_status():
    return {
        "circuit_open": circuit_open,
        "circuit_open_time": circuit_open_time,
        "failure_timestamps": failure_timestamps
    }

async def exponential_backoff_retry(func, *args, **kwargs) -> JSONResponse|HTTPXResponse:
    """Performs a request with exponential backoff on retryable errors."""
    global circuit_open, circuit_open_time, failure_timestamps

    last_response_status_code = 523
    error_response = ""
    async with circuit_lock:
        # Check if the circuit breaker is open
        if circuit_open and time.time() - circuit_open_time < settings.proxy_recovery_time:
            return JSONResponse(content={"error": "Circuit breaker open. Try later."}, status_code=503)

    attempts = 1 + settings.proxy_max_retries
    for attempt in range(attempts):
        try:
            response = await func(*args, **kwargs)
            last_response_status_code = response.status_code
            error_response=extract_content(response)
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
            if settings.proxy_failure_threshold>0 and len(failure_timestamps) >= settings.proxy_failure_threshold:
                circuit_open = True
                circuit_open_time = time.time()
                logger.error("Circuit breaker activated due to multiple failures.", details=error_response)
                return JSONResponse(content={"error": "Circuit breaker activated. Try later."}, status_code=503)

        # Exponential backoff delay
        delay = settings.proxy_base_delay * (settings.proxy_backoff_factor ** attempt) + random.uniform(0, 0.1)
        logger.debug(f"Retrying request in {delay:.2f} seconds.")
        await asyncio.sleep(delay)

    logger.error("Max retries exceeded", details=error_response)
    return JSONResponse(content=error_response, status_code=last_response_status_code)

class RequestStreamWrapper(AsyncByteStream):
    def __init__(self, request):
        self._stream = request.stream()

    async def __aiter__(self) -> AsyncIterator[bytes]:
        async for chunk in self._stream:
            yield chunk

async def stream_mutipart_post(request: Request, client: AsyncClient, target_url: str, headers: dict) -> JSONResponse:
    forwarded_headers = {
        key: value
        for key, value in headers.items()
        if key.lower() not in ("content-length", "transfer-encoding", "connection", "expect", "host")
    }

    # Используем async with для правильного управления генератором потока
    async with client.stream(
            method="POST",
            url=target_url,
            headers=forwarded_headers,
            content=RequestStreamWrapper(request),
    ) as response:
        body = await response.aread()
        try:
            json_data = extract_content(response, True)
        except Exception:
            try: 
                json_data = json.loads(body.decode("utf-8", errors = "replace"))
            except Exception:
                json_data = { "error": "Invalid JSON response"}
        
        return JSONResponse(content=json_data, status_code=response.status_code)

def extract_content(response, raiseException = False) -> dict:
    if isinstance(response, HTTPXResponse):
        try:
            return response.json()
        except Exception:
            try:
                return {"detail": response.text}
            except Exception as e:
                if raiseException:
                    raise
                return {"detail": f"Unable to parse httpx response: {str(e)}"}
    
    if isinstance(response, JSONResponse):
        try:
            if not getattr(response, "body", None):
                response.render()
            return json.loads(response.body.decode("utf-8", errors="replace"))
        except Exception as e:
            if raiseException:
                    raise
            return {"detail": f"Unable to parse JSONResponse: {str(e)}"}
    return {"detail": "Unknown response object type"}

async def get_model_version(model_name: str, client: AsyncClient, request: Request, custom_headers: dict[str, str] = {}):
    path = "v1/chat/completions"
    target_url = f"{settings.proxy_target_url}/{path}"
    body = {
                "model": model_name,
                "messages": [{ "role": "user", "content": "Пришли в ответ любую 1 цифру"}],
                "stream": False,
                "update_interval": 0
            } 
    try:
        response = await exponential_backoff_retry(
            client.request, "POST", target_url, headers=custom_headers, json=body
        )
        if isinstance(response, HTTPXResponse) and response.status_code in {200, 201, 202, 203, 204, 205, 206, 207, 208, 226}:
            current_model = response.json().get("model")
            if current_model:
                name = current_model.split(":")[0]
                version = current_model.split(":")[1]
                if name.endswith("-Pro"):
                    name = name.replace("-Pro","-90b-128k-base")
                elif name.endswith("-Max"):
                    name = name.replace("-Max","-38b-128k-base")
                else:
                    name = f"{name}-9b-128k-base"
                return {"version": f'{name}:{version}'}
        else:
            response_dict = extract_content(response)
            logger.error("Proxy error details", details={
                "status_code": response.status_code,
                "target_url": target_url,
                "response": response_dict,
            })
            error_response = {
                "error": {
                    "status_code": response.status_code,
                    "message": response_dict.get("message", "Proxy request failed"),
                    "details": {
                        "response": response_dict
                    }
                }
            }
            return JSONResponse(content=error_response, status_code=response.status_code)

    except Exception as e:
        logger.error("get_model_version exception", details={
            "target_url": target_url,
            "exception": str(e),
            "exception_type": type(e).__name__
        })
        return JSONResponse(content={
            "error": {
                "status_code": 500,
                "message": "get_model_version request failed",
            }
        }, status_code=500)

async def proxy_request_with_retries(client: AsyncClient, path: str, request: Request, custom_headers: dict[str, str] = {}):
    target_url = f"{settings.proxy_target_url}/{path}"
    if request.url.query:
        target_url = f"{target_url}?{request.url.query}"
    method = request.method
    headers = dict(request.headers)
    logger.debug(f"original headers {redact_headers(headers)}")
    exclude = { h.strip().lower() for h in settings.proxy_exclude_headers.split(",") if h.strip()}
    headers = {k: v for k, v in headers.items() if not any(fnmatch.fnmatch(k.lower(), pattern) for pattern in exclude)}
    logger.debug(f"cleaned headers {redact_headers(headers)}")
    headers.update(custom_headers)  # Inject additional headers
    headers["accept-encoding"]="identity" # Don't accept gzip, br, etc to avoid proxing problem when content decoded but header is
    logger.debug(f"final headers {redact_headers(headers)}")
    #custom_response_headers = {}

    try:
        if method.lower()=="post" and headers.get("content-type", "").lower().startswith("multipart/form-data"):
            response = await stream_mutipart_post(request, client, target_url, headers=headers)
            return response
        else:
            body = await request.body() # BAD solution for multipart request == buffer full body in memory
            response = await exponential_backoff_retry(
                client.request, method, target_url, headers=headers, content=body
            )

        if (isinstance(response, HTTPXResponse)) and response.status_code in {200, 201, 202, 203, 204, 205, 206, 207, 208, 226}:
            logger.debug(f"Proxy request successful: {method} {target_url} -> {response.status_code}")

            if 'completions' in path:
                metadata = ChainMetadataForTracking(
                    chain_type=ChainType.prompt, 
                    chain_name="proxy", 
                    group_id=request.headers.get("x-group-id", "unknown"))
                llm_usage_metrics_handler = MetricsCallbackHandler(metadata)
                try:
                    llm_usage_metrics_handler.on_llm_end(json.loads(response.text))
                except Exception as e:
                    logger.error("Error processing LLM usage metrics", exc_info=e)
            # # Create a new response with the correct headers
            # response_headers = {key: value for key, value in response.headers.items()}
            # response_headers.update(custom_response_headers)  # Inject custom headers
            return JSONResponse(
                    content= extract_content(response),
                    status_code=response.status_code,
                    #headers=response_headers
                )

        else:
            response_dict = extract_content(response)
            # Log full details (with redacted headers) for debugging
            logger.error("Proxy error details", details={
                "status_code": response.status_code,
                "target_url": target_url,
                "method": method,
                "headers": redact_headers(headers),
                "response": response_dict,
                "circuit_breaker_state": "open" if response.status_code == 503 else "unknown",
            })
            # Return sanitized error to client — no headers, no internal URLs
            error_response = {
                "error": {
                    "status_code": response.status_code,
                    "message": response_dict.get("message", "Proxy request failed"),
                    "details": {
                        "response": response_dict,
                        "circuit_breaker_state": "open" if response.status_code == 503 else "unknown",
                    }
                }
            }
            return JSONResponse(content=error_response, status_code=response.status_code)

    except Exception as e:
        # Log full details for debugging
        logger.error("Proxy exception", details={
            "target_url": target_url,
            "method": method,
            "exception": str(e),
            "exception_type": type(e).__name__
        })
        # Return sanitized error to client — no internal URLs or exception details
        return JSONResponse(content={
            "error": {
                "status_code": 500,
                "message": "Proxy request failed",
            }
        }, status_code=500)
