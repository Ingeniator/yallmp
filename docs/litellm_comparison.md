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

## Enterprise-only LiteLLM features that yallmp includes for free

LiteLLM open-source lacks several governance features that only ship in its paid Enterprise tier. yallmp includes equivalents of these at no license cost:

| Feature | yallmp | LiteLLM OSS | LiteLLM Enterprise |
|---|---|---|---|
| Org/team spend hierarchy + budgets | `BillingMiddleware` enforces per-org and per-user Redis-backed limits (`app/middlewares/billing_middleware.py`) via `x-group-id: org/user` | Per-key/user budgets only | Multi-tenant architecture, tag-based budgets, model-specific budgets per virtual key |
| Role-scoped dashboard views | `app/services/dashboard.py` filters metrics by `group_id`/org-prefix based on `x-role` (`ORG_ADMIN`, super-admin) | Single global view | Role-based access control (orgs, teams, user roles) |
| Team-scoped observability | Dashboard/Prometheus queries scoped to caller's group or org | Logging is global | Team-based logging, disable logging per team |
| Spend reporting by group | Built into the same dashboard aggregation | Not in OSS | Generate spend reports |

**Caveat**: yallmp's scoping is authorization-by-trusted-header, not authentication. `x-group-id`/`x-role` (see `app/core/security.py::sanitize_group_id`) are read straight off request headers with no signature or identity verification — something upstream (a gateway or auth proxy) is assumed to have authenticated the caller and set those headers correctly. LiteLLM Enterprise's RBAC comes with an actual SSO/SAML/JWT identity layer attached; yallmp gives you the segmentation and billing logic but not the identity verification, so that piece has to be supplied in front of it.

Other Enterprise-only LiteLLM features yallmp doesn't attempt to replicate: SSO/SAML/SCIM, audit logs with retention, IP allowlisting, secret-manager integrations (Vault/KMS), per-team guardrail assignment, custom UI/email branding, multi-region/admin-worker split architecture.
