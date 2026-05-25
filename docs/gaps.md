# Tracing Gaps

Audit of attributes missing from traces sent to Langfuse.
Every trace should store: **prompts, tools, retrieved docs, outputs, latency, cost, user feedback**.

---

## Status Overview

| # | Attribute       | Status        |
|---|-----------------|---------------|
| 1 | prompts         | ⚠️ Conditional |
| 2 | tools           | ❌ Missing     |
| 3 | retrieved docs  | ❌ Missing     |
| 4 | outputs         | ✅ Fixed (streaming) / ⚠️ Conditional (tracing_log_io gate) |
| 5 | latency         | ⚠️ Partial     |
| 6 | cost            | ⚠️ Conditional |
| 7 | user feedback   | ❌ Missing     |

---

## Gap Details

### 1. Prompts — ⚠️ Conditional
`input=input_body` is only set when `TRACING_LOG_IO=true`.  
When `tracing_log_io=False`, `tracing.py:73-75` nulls out both `input_body` and `output_body`
before the emitter call — prompts silently disappear.

**Fix:** Either always send prompts regardless of `tracing_log_io`, or introduce a separate
`tracing_log_prompts` flag so prompts and raw I/O can be controlled independently.

---

### 2. Tools — ❌ Missing
Tool definitions (`input_body["tools"]`) and tool calls in the response
(`choices[0].message.tool_calls`) are never extracted into dedicated fields.
They may be buried inside `input_body` / `output_body` when `tracing_log_io=True`,
but are invisible otherwise and never structured.

**Fix:** Extract `input_body.get("tools")` and response tool calls into
`metadata["tools_defined"]` / `metadata["tool_calls"]`, or create a child observation
of type `tool` per invocation.

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

### 5. Latency — ⚠️ Partial
`duration_ms` is computed correctly and stored in `metadata["duration_ms"]`.
However, the Langfuse observation is created with `with client.start_as_current_observation(...): pass` —
the span's own `start_time`/`end_time` record ~0 ms because the context manager body is empty.
Langfuse's native timeline view will show the generation as instantaneous.

**Fix:** Pass `start_time` and `end_time` (or `completion_start_time`) directly to
`start_as_current_observation` so the span duration matches the real request latency.

---

### 6. Cost — ⚠️ Conditional
`cost_details` is only populated when both `provider_prefix` and `pricing_cache` are resolved.
In the single-provider path without a pricing cache, or when the model is not found in the
cache, `cost=None` and nothing is written.

**Fix:** Before giving up, fall back to `pricing_cache.find_cost()` (which already searches
across all providers). Log a warning when cost cannot be resolved so misconfiguration is visible.

---

### 7. User Feedback — ❌ Missing
No feedback mechanism exists anywhere in the codebase:
- No `POST /v1/feedback` endpoint.
- No `submit_feedback` / `score` method on `LangfuseEmitter`.
- No call to `langfuse.score(...)` at any point.

**Fix:**
1. Add a `score(trace_id, name, value, comment)` method to `LangfuseEmitter`
   (calls `client.score(...)`).
2. Expose a `POST /v1/feedback` endpoint that accepts `{ trace_id, score, comment }`
   and calls the emitter's `score` method.

---

## Relevant Files

| File | Role |
|------|------|
| `app/services/langfuse_tracing.py` | `LangfuseEmitter` — the actual Langfuse calls |
| `app/services/tracing.py` | Facade + `tracing_log_io` gate |
| `app/core/proxy.py` | `_emit_completions_metrics`, `_emit_streaming_metrics` — call sites |
