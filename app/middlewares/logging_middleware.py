from app.core.logging_config import setup_logging
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
        start_time = time.time()
        
        # Read request body (only for debugging purposes)
        body = await request.body()
        body_str = body.decode("utf-8") if body else None

        # Logging request details
        log_data = {
            "method": request.method,
            "path": request.url.path,
            "query_params": dict(request.query_params),
            "headers": dict(request.headers),
        }
        
        if body_str and logger.isEnabledFor(logging.DEBUG):
            log_data["body"] = body_str  # Log body in DEBUG mode

        logger.info("Incoming Request", **log_data)

        # Process the request
        response = await call_next(request)
        
        # Read response body (only if debugging)
        response_body = b""
        async for chunk in response.body_iterator:
            response_body += chunk
        
        # Clone the response (because body can be consumed only once)
        response = Response(
            content=response_body, 
            status_code=response.status_code,
            headers=dict(response.headers),
            media_type=response.media_type
        )

        # Logging response details
        process_time = time.time() - start_time
        response_log = {
            "status_code": response.status_code,
            "headers": dict(response.headers),
            "process_time": f"{process_time:.4f}s",
        }
        
        if logger.isEnabledFor(logging.DEBUG):
            try:
                response_log["body"] = json.loads(response_body.decode("utf-8"))
            except ValueError:
                response_log["body"] = response_body.decode("utf-8")  # Log as raw text

        logger.info("Outgoing Response", **response_log)

        return response
