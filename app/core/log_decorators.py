import structlog
import functools
import asyncio

logger = structlog.get_logger()


def log_context(**context):
    """Decorator to temporarily bind static context to logs within a function.

    Usage:
    @log_context(user_id=123, action="processing")
    def my_function():
        logger.info("Function called")"""
    def decorator(func):
        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                with logger.bind(**context):
                    return await func(*args, **kwargs)
            return async_wrapper
        else:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                with logger.bind(**context):
                    return func(*args, **kwargs)
            return wrapper
    return decorator


def log_dynamic_context(*context_keys):
    """Decorator to bind function arguments as log context.

    Usage:
    @log_dynamic_context("user_id", "action")
    def my_function(user_id, action, ...):
        logger.info("Function called")"""
    def decorator(func):
        def _build_context(args, kwargs):
            return {
                key: kwargs.get(key, args[i]) if i < len(args) else kwargs.get(key)
                for i, key in enumerate(context_keys)
                if i < len(args) or key in kwargs
            }

        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                with logger.bind(**_build_context(args, kwargs)):
                    return await func(*args, **kwargs)
            return async_wrapper
        else:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                with logger.bind(**_build_context(args, kwargs)):
                    return func(*args, **kwargs)
            return wrapper
    return decorator


def log_with_context_and_exceptions(**context):
    """Decorator to bind context and log exceptions inside a function.

    Usage:
    @log_with_context_and_exceptions(user_id=123, action="processing")
    def my_function():
        logger.info("Function called")
        raise ValueError("Database error!")"""
    def decorator(func):
        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                with logger.bind(**context):
                    try:
                        return await func(*args, **kwargs)
                    except Exception as e:
                        logger.exception("Error in function", error=str(e))
                        raise
            return async_wrapper
        else:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                with logger.bind(**context):
                    try:
                        return func(*args, **kwargs)
                    except Exception as e:
                        logger.exception("Error in function", error=str(e))
                        raise
            return wrapper
    return decorator
