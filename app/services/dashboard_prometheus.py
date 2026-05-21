import asyncio
import logging

import httpx

logger = logging.getLogger(__name__)


def _build_group_filter(group_id, is_org_admin, is_super_admin=False):
    """Build a PromQL label matcher for group_id filtering.

    - is_super_admin → "" (no filter, sees all spaces)
    - None → "" (no filter)
    - is_org_admin with org prefix → group_id=~"org/.*"
    - exact → group_id="val"
    """
    if is_super_admin:
        return ""
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


async def _query_prometheus(client, url, query, eval_time=None):
    """Execute a PromQL instant query against the Prometheus HTTP API.

    Args:
        eval_time: Optional unix timestamp to evaluate the query at (for custom ranges).

    Returns the result list or empty list on error.
    """
    try:
        params = {"query": query}
        if eval_time is not None:
            params["time"] = eval_time
        resp = await client.post(
            f"{url}/api/v1/query",
            data=params,
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


async def fetch_metrics_from_prometheus(url, timeout, group_id, is_org_admin, endpoint_patterns, auth=None, verify=True, time_window="", eval_time=None, is_super_admin=False):
    """Query Prometheus HTTP API for all dashboard metrics.

    Returns a dict matching the shape of parse_metrics_to_dict:
    {"token_usage": [...], "http_requests": [...], "http_duration": [...], "cost": [...]}

    When time_window is set (e.g. "1d", "7d"), counters are wrapped with
    increase(metric[window]) to return the delta over that period.
    """
    group_filter = _build_group_filter(group_id, is_org_admin, is_super_admin)
    endpoint_filter = _build_endpoint_filter(endpoint_patterns)
    token_selector = _build_selector(group_filter)
    http_selector = _build_selector(group_filter, endpoint_filter)

    if time_window:
        queries = {
            "total_tokens": f"increase(llm_total_token_usage_total{token_selector}[{time_window}])",
            "prompt_tokens": f"increase(llm_prompt_token_usage_total{token_selector}[{time_window}])",
            "completion_tokens": f"increase(llm_completion_token_usage_total{token_selector}[{time_window}])",
            "cost": f"increase(llm_cost_total{token_selector}[{time_window}])",
            "http_requests": f"increase(http_requests_total{http_selector}[{time_window}])",
            "http_duration_sum": f"increase(http_request_duration_seconds_sum{http_selector}[{time_window}])",
            "http_duration_count": f"increase(http_request_duration_seconds_count{http_selector}[{time_window}])",
        }
    else:
        queries = {
            "total_tokens": f"llm_total_token_usage_total{token_selector}",
            "prompt_tokens": f"llm_prompt_token_usage_total{token_selector}",
            "completion_tokens": f"llm_completion_token_usage_total{token_selector}",
            "cost": f"llm_cost_total{token_selector}",
            "http_requests": f"http_requests_total{http_selector}",
            "http_duration_sum": f"http_request_duration_seconds_sum{http_selector}",
            "http_duration_count": f"http_request_duration_seconds_count{http_selector}",
        }

    async with httpx.AsyncClient(timeout=timeout, auth=auth, verify=verify) as client:
        results = await asyncio.gather(
            *[_query_prometheus(client, url, q, eval_time=eval_time) for q in queries.values()]
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


async def _query_prometheus_range(client, url, query, start, end, step):
    """Execute a PromQL range query. Returns result list or empty on error."""
    try:
        resp = await client.post(
            f"{url}/api/v1/query_range",
            data={"query": query, "start": str(start), "end": str(end), "step": str(step)},
        )
        if resp.status_code != 200:
            logger.warning("Prometheus range query failed (HTTP %s): %s", resp.status_code, query)
            return []
        data = resp.json()
        if data.get("status") != "success":
            logger.warning("Prometheus range query non-success: %s", data.get("error", ""))
            return []
        return data["data"]["result"]
    except Exception as exc:
        logger.warning("Prometheus range query error: %s", exc)
        return []


_TREND_WINDOWS = {
    "1h":  (3600,    300),    # 5 min step → ~12 points
    "1d":  (86400,   3600),   # 1 h step   → 24 points
    "7d":  (604800,  21600),  # 6 h step   → 28 points
    "30d": (2592000, 86400),  # 1 d step   → 30 points
}


async def fetch_cost_trends(
    url, timeout, group_id, is_org_admin, is_super_admin=False, auth=None, verify=True,
    time_window="7d", group_by="group_id", start="", end="",
):
    """Fetch time-series cost data for line chart.

    Returns {"available": True, "labels": [...], "series": [{"name": ..., "data": [...]}]}
    """
    import time as _time
    from datetime import datetime, timezone

    if start and end:
        try:
            s = datetime.fromisoformat(start).timestamp()
            e = datetime.fromisoformat(end).timestamp()
            duration = e - s
            step = max(int(duration / 30), 60)
        except (ValueError, TypeError):
            return {"available": True, "labels": [], "series": []}
    else:
        duration, step = _TREND_WINDOWS.get(time_window, _TREND_WINDOWS["7d"])
        e = _time.time()
        s = e - duration

    group_filter = _build_group_filter(group_id, is_org_admin, is_super_admin)
    selector = _build_selector(group_filter)
    query = f"sum(increase(llm_cost_total{selector}[{step}s])) by ({group_by})"

    async with httpx.AsyncClient(timeout=timeout, auth=auth, verify=verify) as client:
        results = await _query_prometheus_range(client, url, query, int(s), int(e), step)

    if not results:
        return {"available": True, "labels": [], "series": []}

    all_ts = sorted({v[0] for r in results for v in r.get("values", [])})
    labels = [
        datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%m-%d %H:%M")
        for ts in all_ts
    ]
    series = []
    for r in results:
        name = r["metric"].get(group_by, "unknown")
        ts_to_val = {v[0]: float(v[1]) for v in r.get("values", [])}
        data = [round(ts_to_val.get(ts, 0.0), 6) for ts in all_ts]
        series.append({"name": name, "data": data})

    return {"available": True, "labels": labels, "series": series}
