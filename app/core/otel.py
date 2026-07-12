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
    """Configure the tracer provider and instrument app + httpx. No-op if disabled."""
    global _configured
    if not settings.otel_enabled or _configured:
        return

    resource = Resource.create({"service.name": settings.otel_service_name})
    provider = TracerProvider(resource=resource)
    exporter = OTLPSpanExporter(endpoint=f"{settings.otel_exporter_endpoint.rstrip('/')}/v1/traces")
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)

    FastAPIInstrumentor.instrument_app(app)
    HTTPXClientInstrumentor().instrument()

    logger.info(
        "OpenTelemetry tracing enabled",
        endpoint=settings.otel_exporter_endpoint,
        service_name=settings.otel_service_name,
    )
    _configured = True


def shutdown_otel() -> None:
    """Flush and shut down the tracer provider — called from lifespan."""
    global _configured
    if not _configured:
        return
    provider = trace.get_tracer_provider()
    if hasattr(provider, "shutdown"):
        provider.shutdown()
    _configured = False
