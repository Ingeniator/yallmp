import asyncio
import logging

import httpx

logger = logging.getLogger(__name__)


def _build_group_filter(group_id, is_org_admin):
    """Build a PromQL label matcher for group_id filtering.

    - None → "" (no filter)
    - is_org_admin with org prefix → group_id=~"org/.*"
    - exact → group_id="val"
    """
    if not group_id:
        return ""
    if is_org_admin and "/" in group_id:
        org_prefix = group_id.split("/", 1)[0]
        return f'group_id=~"{org_prefix}/.*"'
    return f'group_id="{group_id}"'


def _build_endpoint_filter(patterns):
    """Combine endpoint regex patterns into a PromQL label matcher.

    Returns e.g. endpoint=~"pat1|pat2" or "" if no patterns.
    """
    if not patterns:
        return ""
    combined = "|".join(patterns)
    return f'endpoint=~"{combined}"'


def _build_selector(group_filter, endpoint_filter=""):
    """Combine filters into a PromQL selector string like {group_id="x",endpoint=~"y"}."""
    parts = [f for f in (group_filter, endpoint_filter) if f]
    if not parts:
        return ""
    return "{" + ",".join(parts) + "}"


async def _query_prometheus(client, url, query):
    """Execute a PromQL instant query against the Prometheus HTTP API.

    Returns the result list or empty list on error.
    """
    try:
        resp = await client.get(
            f"{url}/api/v1/query",
            params={"query": query},
        )
        if resp.status_code != 200:
            logger.warning("Prometheus query failed (HTTP %s): %s", resp.status_code, query)
            return []
        data = resp.json()
        if data.get("status") != "success":
            logger.warning("Prometheus query returned non-success: %s", data.get("error", "unknown"))
            return []
        return data.get("data", {}).get("result", [])
    except (httpx.ConnectError, httpx.TimeoutException) as exc:
        logger.warning("Prometheus connection error for query '%s': %s", query, exc)
        return []


def _extract_metric_entries(results, extra_fields=None):
    """Convert Prometheus query results to list of dicts matching local metric shape.

    Each result has {"metric": {labels...}, "value": [timestamp, "string_value"]}.
    """
    entries = []
    for r in results:
        labels = dict(r.get("metric", {}))
        # Remove __name__ — not needed in our data dicts
        labels.pop("__name__", None)
        value = float(r["value"][1])
        entry = {"value": value, **labels}
        if extra_fields:
            entry.update(extra_fields)
        entries.append(entry)
    return entries


async def fetch_metrics_from_prometheus(url, timeout, group_id, is_org_admin, endpoint_patterns):
    """Query Prometheus HTTP API for all dashboard metrics.

    Returns a dict matching the shape of parse_metrics_to_dict:
    {"token_usage": [...], "http_requests": [...], "http_duration": [...], "cost": [...]}
    """
    group_filter = _build_group_filter(group_id, is_org_admin)
    endpoint_filter = _build_endpoint_filter(endpoint_patterns)
    token_selector = _build_selector(group_filter)
    http_selector = _build_selector(group_filter, endpoint_filter)

    queries = {
        "total_tokens": f"llm_total_token_usage_total{token_selector}",
        "prompt_tokens": f"llm_prompt_token_usage_total{token_selector}",
        "completion_tokens": f"llm_completion_token_usage_total{token_selector}",
        "cost": f"llm_cost_total{token_selector}",
        "http_requests": f"http_requests_total{http_selector}",
        "http_duration_sum": f"http_request_duration_seconds_sum{http_selector}",
        "http_duration_count": f"http_request_duration_seconds_count{http_selector}",
    }

    async with httpx.AsyncClient(timeout=timeout) as client:
        results = await asyncio.gather(
            *[_query_prometheus(client, url, q) for q in queries.values()]
        )

    raw = dict(zip(queries.keys(), results))

    # Build token_usage entries with "metric" field matching local format
    token_usage = (
        _extract_metric_entries(raw["total_tokens"], {"metric": "total"})
        + _extract_metric_entries(raw["prompt_tokens"], {"metric": "prompt"})
        + _extract_metric_entries(raw["completion_tokens"], {"metric": "completion"})
    )

    cost = _extract_metric_entries(raw["cost"])
    http_requests = _extract_metric_entries(raw["http_requests"])

    http_duration = (
        _extract_metric_entries(raw["http_duration_sum"], {"stat": "sum"})
        + _extract_metric_entries(raw["http_duration_count"], {"stat": "count"})
    )

    return {
        "token_usage": token_usage,
        "http_requests": http_requests,
        "http_duration": http_duration,
        "cost": cost,
    }
