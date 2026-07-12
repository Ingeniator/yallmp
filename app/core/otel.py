"""OpenTelemetry distributed tracing setup.

Configures a TracerProvider that exports spans over OTLP/HTTP (e.g. to
Tempo) and instruments FastAPI request handling and outbound httpx calls.
This is independent of the Langfuse-based business tracing in
app.services.tracing — that captures LLM-call semantics, this captures
request-level spans for observability/debugging.
"""

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from app.core.config import settings
from app.core.logging_config import setup_logging

logger = setup_logging()

_configured = False


def setup_otel(app) -> None:
    """Configure the tracer provider and instrument app + httpx. No-op if disabled.

    entrypoint.py calls uvicorn.run("entrypoint:app", ...) — a string target
    makes uvicorn re-import the entrypoint module under a second module
    identity, which re-runs create_app() a second time. The SDK-level setup
    (provider/exporter/httpx patch) must only happen once, but
    FastAPIInstrumentor.instrument_app() has to run for every app instance
    created — otherwise whichever instance uvicorn actually ends up serving
    may be the one that never got instrumented. instrument_app() is
    idempotent per app object, so calling it unconditionally is safe.
    """
    if not settings.otel_enabled:
        return

    global _configured
    if not _configured:
        resource = Resource.create({"service.name": settings.otel_service_name})
        provider = TracerProvider(resource=resource)
        exporter = OTLPSpanExporter(endpoint=f"{settings.otel_exporter_endpoint.rstrip('/')}/v1/traces")
        provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(provider)
        HTTPXClientInstrumentor().instrument()

        logger.info(
            "OpenTelemetry tracing enabled",
            endpoint=settings.otel_exporter_endpoint,
            service_name=settings.otel_service_name,
        )
        _configured = True

    FastAPIInstrumentor.instrument_app(app)


def shutdown_otel() -> None:
    """Flush and shut down the tracer provider — called from lifespan."""
    global _configured
    if not _configured:
        return
    provider = trace.get_tracer_provider()
    if hasattr(provider, "shutdown"):
        provider.shutdown()
    _configured = False
