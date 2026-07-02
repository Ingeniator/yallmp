# Tracing Gaps

Audit of attributes missing from traces sent to Langfuse.
Every trace should store: **prompts, tools, retrieved docs, outputs, latency, cost, user feedback**.

---

## Status Overview

| # | Attribute       | Status        |
|---|-----------------|---------------|
| 1 | prompts         | ⚠️ Conditional |
| 2 | tools           | ✅ Fixed       |
| 3 | retrieved docs  | ❌ Missing     |
| 4 | outputs         | ✅ Fixed (streaming) / ⚠️ Conditional (tracing_log_io gate) |
| 5 | latency         | ✅ Fixed       |
| 6 | cost            | ⚠️ Conditional |
| 7 | user feedback   | ✅ Fixed       |

---

## Gap Details

### 1. Prompts — ⚠️ Conditional
`input=input_body` is only set when `TRACING_LOG_IO=true`.  
When `tracing_log_io=False`, `tracing.py:73-75` nulls out both `input_body` and `output_body`
before the emitter call — prompts silently disappear.

**Fix:** Either always send prompts regardless of `tracing_log_io`, or introduce a separate
`tracing_log_prompts` flag so prompts and raw I/O can be controlled independently.

---

### 2. Tools — ✅ Fixed

Tool names from `input_body["tools"]` are extracted into `metadata["tools_defined"]`
(list of name strings). Tool call names from the response are extracted into
`metadata["tool_calls"]` — from `choices[n].message.tool_calls` for non-streaming,
and from `delta.tool_calls[n].function.name` SSE chunks for streaming.

Both fields are populated independently of `tracing_log_io` and are omitted from
`metadata` when empty (no tools in the request).

---

### 3. Retrieved Docs — ❌ Missing
The proxy is a passthrough and has zero RAG visibility.
Retrieved document chunks are never passed in, and even when embedded in `messages`
they are indistinguishable from normal message content.

**Fix:** Define a convention for clients to forward retrieval context — e.g. an
`x-retrieval-context` JSON header or a structured wrapper in the request body —
and create a `retrieval` child observation before the generation observation.

---

### 4. Outputs — ✅ Streaming fixed / ⚠️ Conditional (`tracing_log_io`)

**Streaming — fixed** (`proxy.py`, `_assemble_streaming_output`).  
`_emit_streaming_metrics` now calls `_assemble_streaming_output(full_text, last_payload)` which
walks every `data:` line, accumulates `choices[n].delta.content` per index, and returns a
`chat.completion`-shaped dict with the full reply text. The last SSE chunk (which had only usage
and an empty delta) is no longer used as the output.

Remaining issue:
- **`tracing_log_io` gate** — `output=output_body` becomes `None` when `tracing_log_io=False`
  (same as prompts gap #1).

---

### 5. Latency — ✅ Fixed

`end_time = datetime.now(timezone.utc)` and `start_time = end_time - timedelta(milliseconds=duration_ms)`
are now computed in `LangfuseEmitter.trace_proxy_request` and passed directly to
`start_as_current_observation`. The Langfuse timeline view now reflects the real request latency.
`duration_ms` is still stored in `metadata["duration_ms"]` for raw querying.

---

### 6. Cost — ⚠️ Conditional
`cost_details` is only populated when both `provider_prefix` and `pricing_cache` are resolved.
In the single-provider path without a pricing cache, or when the model is not found in the
cache, `cost=None` and nothing is written.

**Fix:** Before giving up, fall back to `pricing_cache.find_cost()` (which already searches
across all providers). Log a warning when cost cannot be resolved so misconfiguration is visible.

---

### 7. User Feedback — ✅ Fixed

**`POST /v1/feedback`** accepts `{ request_id, score, name?, comment? }` and attaches a score
to the corresponding Langfuse trace. Group isolation is preserved via `X-Group-ID`.

Trace linkage is stateless: the Langfuse `trace_id` is derived from `request_id` with
`Langfuse.create_trace_id(seed=request_id)` — the same formula used when the original request
was traced — so no mapping needs to be stored between request and feedback.

`LangfuseEmitter.score()` POSTs directly to `/api/public/scores` via `httpx.AsyncClient`
(the OTEL-based Langfuse v3 SDK has no `score` method; the HTTP API is authoritative).
Returns 503 when `tracing_enabled=False`.

---

## Relevant Files

| File | Role |
|------|------|
| `app/services/langfuse_tracing.py` | `LangfuseEmitter` — the actual Langfuse calls |
| `app/services/tracing.py` | Facade + `tracing_log_io` gate |
| `app/core/proxy.py` | `_emit_completions_metrics`, `_emit_streaming_metrics` — call sites |
