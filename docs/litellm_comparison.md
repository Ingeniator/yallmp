# yallmp vs LiteLLM

## What they share

- LLM proxy / unified API endpoint
- Multi-provider routing
- Retry with exponential backoff, circuit breaking / fallbacks
- Rate limiting
- Observability (Prometheus metrics, Langfuse tracing)
- Cost/token tracking

## What LiteLLM adds over yallmp

- 100+ provider integrations out of the box (OpenAI, Anthropic, Azure, Bedrock, Vertex, etc.)
- Caching layer (Redis/in-memory)
- Virtual API keys with per-key budgets and rate limits
- Team/org spend management
- Built-in UI dashboard

## What yallmp adds over LiteLLM

### Prompt Hub (`/prompts`, `/prompt/format/{name}`, `/prompt/execute/{name}`)
Centralized prompt store with templating and variable substitution. LiteLLM has no prompt management concept.

### Chain Hub (`/chains`, `/chain/execute/{name}`)
Execute LangChain chains via HTTP. LiteLLM doesn't touch LangChain.

### Search Hub (`/search`, `/search/providers`, `/search/ui`)
Unified search API routing across Brave, Tavily, and Exa — with the same circuit-breaking, billing, and tracing applied to LLM calls. LiteLLM has no web search routing.

### User Feedback API (`POST /v1/feedback`)
Attach thumbs-up/down scores to past LLM traces by `X-Request-ID`. LiteLLM has no built-in feedback collection endpoint.

### Group/Org-aware billing with sync
Spend limits organized by `x-group-id` (org/user hierarchy), enforced in Redis, with tier config in YAML and periodic sync from an external tracing backend (llogr). LiteLLM has virtual key budgets but no org-tree hierarchy or external sync.

### Integrated dashboard
Built-in web UI at `/dashboard` showing cost/usage trends (Prometheus-backed), session browsing, and trace drilldown — role-gated by `x-role`/`x-group-id` headers. LiteLLM's dashboard is a separate product.

### K8s-native readiness probe
`/ready` reports per-provider circuit breaker status with 503 on degradation, not just process liveness. LiteLLM's health checks are simpler.
