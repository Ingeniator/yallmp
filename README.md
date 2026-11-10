# YALLMP (Yet Another LLM Proxy)

A proxy server for Language Learning Models (LLMs) with built-in circuit breaking, rate limiting, and observability.

## Features

- **Proxy Service**
  - Configurable circuit breaker
  - Exponential backoff retry mechanism

- **Chain Hub**
  - Load and execute LangChain chains

- **Prompt Hub**
  - Centralized prompt management
  - Prompt templating
  - Variable substitution

- **Observability**
  - Health check endpoints
  - Structured logging
  - Request/Response metrics

## Quick Start

### Prerequisites
- Python 3.13+
- Make

### Installation

1. Clone the repository:
```bash
git clone https://github.com/Ingeniator/yallmp.git
cd yallmp
```

2. Install dependencies:
```bash
make dev-init
```

3. Configure environment:
```bash
cp .env.example .env
# Edit .env with your settings
```

### Running the Service

```bash
make run
```

The service will start on http://localhost:8000 by default.

## API Documentation

### Health & Observability

| Endpoint | Purpose | K8s Probe |
|----------|---------|-----------|
| `GET /livez` | Liveness — instant 200, no dependency checks | `livenessProbe` |
| `GET /ready` | Readiness — checks circuit breakers (proxy + per-provider); returns 200 or 503 | `readinessProbe` |
| `GET /health` | Full status — JSON with component statuses, version, and diagnostic details | Dashboards / monitoring |
| `GET /metrics` | Prometheus metrics (request counts, latency histograms, LLM token/cost counters) | — |

### Alerting

Prometheus alerting rules and Alertmanager config are in `devops/alerting/`:

- `alert_rules.yml` — error rate spikes, circuit breaker open, P95/P99 latency, LLM cost anomaly, health check failures
- `alertmanager.yml` — routing, receivers, cross-service inhibition rules
- `prometheus.yml` — scrape config for all ai-suite services

### Core Endpoints

- `POST|GET /llm/{path}` - Proxy LLM requests to a provider
- `POST|GET /llm/@{prefix}/{path}` - Proxy to the provider with that prefix
- `GET /llm/v1/models` - Models of all providers, as `<prefix>/<model>`

### Providers and routing

Set `LLM_PROXY_ENABLED=true` and put one JSON file per upstream in
`llm_hub_directory` (default `data/llm_hub`). There is no separate single-upstream
mode: the old `LLM_PROXY_TARGET_URL` / `LLM_PROXY_AUTHORIZATION_TYPE` / `LLM_PROXY_PRICING_*`
settings are ignored (a startup warning lists any still set). `LLM_LLM_HUB_ENABLED`
is a deprecated alias of `LLM_PROXY_ENABLED`.

A request can pin a provider explicitly. This is the way to reach endpoints that
carry no model, such as files, fine-tuning jobs and batches:

- **Path:** `/llm/@<prefix>/v1/...`, e.g. `/llm/@openai/v1/files`. With an SDK, set
  `base_url=".../llm/@openai/v1"` and every call (uploads, job status, completions)
  goes to that provider.
- **Header:** `X-Provider: <prefix>` on a normal `/llm/v1/...` path. An unknown
  prefix gets 404; a header that conflicts with the path prefix gets 400. The
  header is not forwarded upstream. It also applies to `/llm/v1/models` (the
  provider's own list, unprefixed) and `/llm/version`.

A pin is a hard guarantee: the request never leaves that provider, which keeps
files, fine-tuned models and billing on one upstream. Model routing is skipped:

- An alias uses only its `target`/`fallback` entries that resolve to the pinned
  provider (the fallback still applies if both are there). Entries on other
  providers are ignored, so the same alias can fail when pinned where it would
  fall back to another provider unpinned.
- An alias with no entry on the pinned provider, or any other name, is sent as is,
  with only the provider's own `<prefix>/` stripped. Alias names may be real model
  names upstream.
The `@` marks a provider: upstream API paths never start with it, so `/llm/openai/...`
(no `@`) is still an ordinary upstream path for the default provider. An unknown
`@<prefix>` gets 404. Prefixes can't contain `/`.

```python
client = OpenAI(base_url="http://yallmp/ai/llm/@openai/v1", api_key="...")
f = client.files.create(file=open("train.jsonl", "rb"), purpose="fine-tune")
job = client.fine_tuning.jobs.create(training_file=f.id, model="gpt-4o-mini")
client.fine_tuning.jobs.retrieve(job.id)
```

Without a pin, the `model` in the request body picks the provider:

1. An alias from `aliases.json`, resolved to its `target` (and `fallback` on error).
2. `<prefix>/<model>` with a known prefix: that provider, prefix stripped.
3. A name in some provider's `models` list: that provider, name unchanged.
4. Anything else: the provider with `"default": true`, name unchanged.
   Without a default provider such requests get 404.

Unpinned requests without a model (GET, multipart uploads) go to the default provider.

```json
{
    "prefix": "localllm",
    "base_url": "${LLM_LOCALLLM_BASE_URL}",
    "default": true,
    "auth": { "type": "NONE" }
}
```

A provider that references an unset `${VAR}` is skipped, so environment variables
decide which providers load in each deployment.

### Request Headers

Clients calling the proxy (`/llm/...`) can pass these optional headers to control billing grouping and tracing. All are optional — omitting them just means the corresponding trace/billing field is left blank or defaulted.

| Header | Purpose | Notes |
|--------|---------|-------|
| `X-Group-Id` | Billing/rate-limit grouping key | Format `org/user` (e.g. `acme/alice`). The part before `/` is billed as the org; without a `/` the whole value is treated as an org with no per-user tracking. Defaults to `unknown`. |
| `X-Session-Id` | Groups requests into a Langfuse session | Free-form string. |
| `X-Request-Id` | Client-supplied trace id | Used as the seed for the Langfuse trace id, so repeated calls with the same value stitch into one trace. |
| `X-Agent-Name` | Name of the calling agent/app | Recorded in trace metadata (`agent_name`) and added as a trace tag. If omitted, falls back to sniffing the standard `User-Agent` header — see note below. |
| `X-Tags` | Custom trace tags | Comma-separated list, e.g. `X-Tags: eval, nightly-run`. Merged with the auto-generated tags (provider, agent name, request id). |
| `X-Prompt-Name` | Name of the prompt template used | Recorded on the trace observation. |
| `X-Prompt-Version` | Version of the prompt template used | Recorded on the trace observation. |

**`User-Agent` fallback for `agent_name`:** When `X-Agent-Name` isn't set, the proxy tries
to detect the calling CLI/SDK from the standard `User-Agent` header instead (see
`_CLI_AGENT_USER_AGENT_PATTERNS` in `app/core/proxy.py` — covers signatures like
`claude-code`, `aider`, `cursor`, `continue`, `cline`, `windsurf`, `codex-cli`, `litellm`,
`langchain`, `llamaindex`, `openai-sdk`, `anthropic-sdk`). This is a curated allow-list, not
a generic "first token of User-Agent" extraction — an unrecognized `User-Agent` yields no
`agent_name` at all, by design:

- **Cardinality** — `agent_name` feeds directly into trace `tags`. Passing through raw
  User-Agent strings verbatim (which often embed exact version numbers) would turn every
  client version into its own permanent tag.
- **Signal quality** — generic HTTP libraries (`curl`, `python-requests`, `axios`, ...) would
  get labeled as "agents" even though they're just the transport, muddying "which coding
  agent called us" queries.
- **Spoofability** — `User-Agent` is entirely client-controlled, so pass-through adds no
  security benefit, only uncurated tag noise.

If you need visibility into an agent that isn't in the list yet, either add its signature to
`_CLI_AGENT_USER_AGENT_PATTERNS` or have that client send `X-Agent-Name` explicitly.

Example:

```bash
curl -X POST http://localhost:8888/ai/llm/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'X-Group-Id: acme/alice' \
  -H 'X-Agent-Name: checkout-bot' \
  -H 'X-Tags: eval, nightly-run' \
  -d '{"model": "llama-3.2-3b-instruct", "messages": [{"role": "user", "content": "hi"}]}'
```

### Chain Hub

- `POST /chain/{name}` - Execute a chain
- `GET /chains` - List available chains

### Prompt Hub

- `POST /prompt/{name}` - Format a prompt
- `GET /prompts` - List available prompts

## Development

### Setup
```bash
make dev-init
```

### Running
```bash
make run
```

### Running Tests

```bash
pytest
```

### Code Quality

```bash
# Format code and lint
make lint
```

## Contributing

1. Fork the repository
2. Create your feature branch
3. Run tests and linting
4. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
