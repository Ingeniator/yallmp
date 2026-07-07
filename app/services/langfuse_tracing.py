from __future__ import annotations

import json
from collections import OrderedDict
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING

import opentelemetry.trace as otel_trace_api
from langfuse import Langfuse, LangfuseGeneration, LangfuseSpan
from langfuse._client.attributes import LangfuseOtelSpanAttributes

from app.core.config import settings
from app.core.logging_config import setup_logging

if TYPE_CHECKING:
    from app.services.pricing import CostBreakdown

# Langfuse internal attribute: tells the backend to render this span as a root trace
# even though it carries a remote parent context (needed for trace_id stitching).
_LANGFUSE_AS_ROOT_ATTR = "langfuse.internal.as_root"

logger = setup_logging()

MAX_CLIENT_CACHE_SIZE = 128


class LangfuseEmitter:
    """TraceEmitter implementation backed by Langfuse v3 SDK.

    Uses the native OTEL-based Langfuse client which sends spans
    to /api/public/otel/v1/traces via protobuf.

    Per-group isolation: each unique group_id gets its own Langfuse client
    with public_key=group_id, so the backend can separate logs by group.
    """

    def __init__(self):
        from langfuse import Langfuse

        self._host = settings.tracing_host
        self._public_key = settings.tracing_public_key
        self._secret_key = settings.tracing_secret_key
        self._environment = settings.tracing_environment
        self._release = settings.tracing_release
        self._default_client = Langfuse(
            host=self._host,
            public_key=self._public_key,
            secret_key=self._secret_key,
            environment=self._environment,
            release=self._release,
        )
        self._clients: OrderedDict[str, object] = OrderedDict()

        from importlib.metadata import version as pkg_version
        langfuse_ver = pkg_version("langfuse")
        logger.info("Langfuse client initialized (OTEL mode)", host=self._host, langfuse_version=langfuse_ver)

    def _get_client(self, group_id: str):
        if not group_id or group_id == "unknown":
            return self._default_client
        if group_id in self._clients:
            self._clients.move_to_end(group_id)
            return self._clients[group_id]

        from langfuse import Langfuse

        client = Langfuse(
            host=self._host,
            public_key=group_id,
            secret_key=group_id,
            environment=self._environment,
            release=self._release,
        )
        self._clients[group_id] = client

        # Evict least-recently-used client if cache is full
        if len(self._clients) > MAX_CLIENT_CACHE_SIZE:
            _, evicted = self._clients.popitem(last=False)
            evicted.flush()
            evicted.shutdown()

        return client

    def trace_proxy_request(
        self,
        model: str,
        provider: str | None,
        input_body: dict | None,
        output_body: dict | None,
        status_code: int,
        usage: dict | None,
        duration_ms: float,
        group_id: str,
        is_streaming: bool,
        cost: "CostBreakdown | None" = None,
        session_id: str | None = None,
        trace_id: str | None = None,
        tools_defined: list[str] | None = None,
        tool_calls: list[str] | None = None,
        agent_name: str | None = None,
        completion_start_time: datetime | None = None,
        prompt_name: str | None = None,
        prompt_version: str | None = None,
    ) -> None:
        client = self._get_client(group_id)

        usage_details = {}
        if usage:
            # OpenAI field names
            if "prompt_tokens" in usage:
                usage_details["input"] = usage["prompt_tokens"]
            if "completion_tokens" in usage:
                usage_details["output"] = usage["completion_tokens"]
            if "total_tokens" in usage:
                usage_details["total"] = usage["total_tokens"]
            # Anthropic native field names (fallback when OpenAI names absent)
            if "input_tokens" in usage and "input" not in usage_details:
                usage_details["input"] = usage["input_tokens"]
            if "output_tokens" in usage and "output" not in usage_details:
                usage_details["output"] = usage["output_tokens"]
            # Anthropic prompt caching
            if "cache_creation_input_tokens" in usage:
                usage_details["cache_creation_input"] = usage["cache_creation_input_tokens"]
            if "cache_read_input_tokens" in usage:
                usage_details["cache_read_input"] = usage["cache_read_input_tokens"]
            # OpenAI o1/o3 reasoning tokens
            reasoning = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")
            if reasoning:
                usage_details["reasoning"] = reasoning

        metadata = {
            "provider": provider,
            "group_id": group_id,
            "is_streaming": is_streaming,
            "status_code": status_code,
            "duration_ms": duration_ms,
        }
        if agent_name:
            metadata["agent_name"] = agent_name
        if tools_defined:
            metadata["tools_defined"] = tools_defined
        if tool_calls:
            metadata["tool_calls"] = tool_calls
        cost_details = None
        if cost is not None:
            cost_details = {"input": cost.input, "output": cost.output, "total": cost.total}
            metadata["cost"] = cost.total
            metadata["input_cost"] = cost.input
            metadata["output_cost"] = cost.output

        model_parameters: dict | None = None
        if input_body:
            _param_keys = ("temperature", "max_tokens", "top_p", "frequency_penalty", "presence_penalty", "seed")
            extracted = {k: input_body[k] for k in _param_keys if k in input_body}
            if extracted:
                model_parameters = extracted

        # Anthropic extended thinking: split thinking blocks out of the output
        # so Langfuse shows the reasoning separately in metadata.
        content_blocks = (output_body or {}).get("content")
        if isinstance(content_blocks, list) and any(b.get("type") == "thinking" for b in content_blocks if isinstance(b, dict)):
            thinking_texts = [b["thinking"] for b in content_blocks if isinstance(b, dict) and b.get("type") == "thinking"]
            metadata["thinking"] = thinking_texts if len(thinking_texts) > 1 else thinking_texts[0]

        level = "ERROR" if status_code >= 400 else None
        status_message = f"HTTP {status_code}" if status_code >= 400 else None

        user_id: str | None = None
        if group_id and "/" in group_id:
            user_id = group_id.split("/", 1)[1] or None

        tags = [t for t in (provider, agent_name) if t] or None

        trace_name = model or "llm-proxy"
        valid_trace_id = Langfuse.create_trace_id(seed=trace_id) if trace_id else None

        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(milliseconds=duration_ms)
        # OTEL span timestamps are nanoseconds since epoch
        start_time_ns = int(start_time.timestamp() * 1e9)
        end_time_ns = int(end_time.timestamp() * 1e9)

        # When we have a trace_id we need to stitch this span into the right Langfuse
        # trace.  Langfuse does this by creating a NonRecordingSpan carrying the desired
        # trace_id, then using it as the OTEL parent context.  We replicate that here so
        # we can still pass start_time/end_time to the raw OTEL tracer — the Langfuse v4
        # public API (start_as_current_observation) no longer exposes those params.
        otel_context = None
        if valid_trace_id:
            remote_parent = client._create_remote_parent_span(
                trace_id=valid_trace_id, parent_span_id=None
            )
            otel_context = otel_trace_api.set_span_in_context(remote_parent)

        with client._otel_tracer.start_as_current_span(
            name=trace_name,
            context=otel_context,   # None → use ambient OTEL context
            start_time=start_time_ns,
            end_on_exit=False,      # we call end() ourselves with the correct ns
        ) as otel_span:
            if valid_trace_id:
                # Mirror what Langfuse sets in _create_span_with_parent_context so the
                # backend renders this as a root-level trace, not a dangling child.
                otel_span.set_attribute(_LANGFUSE_AS_ROOT_ATTR, True)
            # Set directly on the span rather than via propagate_attributes: that
            # helper only cascades through the *ambient* OTEL context, which gets
            # replaced wholesale above whenever we stitch onto an explicit trace_id.
            if session_id:
                otel_span.set_attribute(LangfuseOtelSpanAttributes.TRACE_SESSION_ID, session_id)
            if user_id:
                otel_span.set_attribute(LangfuseOtelSpanAttributes.TRACE_USER_ID, user_id)
            if tags:
                otel_span.set_attribute(LangfuseOtelSpanAttributes.TRACE_TAGS, tags)
            if prompt_name:
                otel_span.set_attribute(LangfuseOtelSpanAttributes.OBSERVATION_PROMPT_NAME, prompt_name)
            if prompt_version:
                otel_span.set_attribute(LangfuseOtelSpanAttributes.OBSERVATION_PROMPT_VERSION, str(prompt_version))
            gen = LangfuseGeneration(
                otel_span=otel_span,
                langfuse_client=client,
                environment=getattr(client, "_environment", None),
                release=getattr(client, "_release", None),
                input=input_body,
                output=output_body,
                metadata=metadata,
                usage_details=usage_details or None,
                cost_details=cost_details,
                completion_start_time=completion_start_time,
                model=model,
                model_parameters=model_parameters,
                level=level,
                status_message=status_message,
            )
            gen.end(end_time=end_time_ns)  # → otel_span.end(end_time=ns) internally

    def trace_search_request(
        self,
        provider: str,
        query: str,
        num_results: int,
        result_count: int,
        status_code: int,
        duration_ms: float,
        group_id: str,
        cost: float | None = None,
        trace_id: str | None = None,
        session_id: str | None = None,
    ) -> None:
        client = self._get_client(group_id)

        metadata = {
            "provider": provider,
            "group_id": group_id,
            "status_code": status_code,
            "duration_ms": duration_ms,
            "num_results_requested": num_results,
            "num_results_returned": result_count,
        }
        cost_details = None
        if cost is not None:
            cost_details = {"total": cost}
            metadata["cost"] = cost

        valid_trace_id = Langfuse.create_trace_id(seed=trace_id) if trace_id else None

        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(milliseconds=duration_ms)
        start_time_ns = int(start_time.timestamp() * 1e9)
        end_time_ns = int(end_time.timestamp() * 1e9)

        otel_context = None
        if valid_trace_id:
            remote_parent = client._create_remote_parent_span(
                trace_id=valid_trace_id, parent_span_id=None
            )
            otel_context = otel_trace_api.set_span_in_context(remote_parent)

        with client._otel_tracer.start_as_current_span(
            name=f"search:{provider}",
            context=otel_context,
            start_time=start_time_ns,
            end_on_exit=False,
        ) as otel_span:
            if valid_trace_id:
                otel_span.set_attribute(_LANGFUSE_AS_ROOT_ATTR, True)
            # Set directly on the span — propagate_attributes only cascades through the
            # ambient OTEL context, which gets replaced wholesale above whenever we
            # stitch onto an explicit trace_id.
            if session_id:
                otel_span.set_attribute(LangfuseOtelSpanAttributes.TRACE_SESSION_ID, session_id)
            span = LangfuseSpan(
                otel_span=otel_span,
                langfuse_client=client,
                environment=getattr(client, "_environment", None),
                release=getattr(client, "_release", None),
                input={"query": query} if query is not None else None,
                output={"result_count": result_count},
                metadata=metadata,
            )
            # LangfuseSpan constructor doesn't accept usage_details / cost_details
            # (those are generation-only params). Set the underlying OTEL attributes
            # directly — same JSON encoding that create_generation_attributes uses.
            otel_span.set_attribute(
                LangfuseOtelSpanAttributes.OBSERVATION_USAGE_DETAILS,
                json.dumps({"total": 1, "unit": "SEARCHES"}),
            )
            if cost_details:
                otel_span.set_attribute(
                    LangfuseOtelSpanAttributes.OBSERVATION_COST_DETAILS,
                    json.dumps(cost_details),
                )
            span.end(end_time=end_time_ns)

    async def score(
        self,
        trace_id: str,
        name: str,
        value: float,
        comment: str | None,
        group_id: str,
    ) -> None:
        """Submit a score via the Langfuse REST API (POST /api/public/scores)."""
        import httpx

        pk = group_id if group_id and group_id != "unknown" else self._public_key
        sk = group_id if group_id and group_id != "unknown" else self._secret_key

        payload: dict = {"traceId": trace_id, "name": name, "value": value}
        if comment:
            payload["comment"] = comment

        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(
                f"{self._host.rstrip('/')}/api/public/scores",
                json=payload,
                auth=(pk, sk),
            )
            resp.raise_for_status()

    def get_langchain_callback(self, trace_name: str, metadata: dict) -> object | None:
        try:
            from langfuse.langchain import CallbackHandler

            group_id = metadata.get("group_id", "")
            client = self._get_client(group_id)
            return CallbackHandler(
                trace_name=trace_name,
                metadata=metadata,
                langfuse_client=client,
            )
        except Exception as e:
            logger.error("Failed to create Langfuse callback handler", exc_info=e)
            return None

    def shutdown(self) -> None:
        self._default_client.flush()
        self._default_client.shutdown()
        for client in self._clients.values():
            client.flush()
            client.shutdown()
        self._clients.clear()
