FROM python:3.11-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY . .

FROM python:3.11-slim

WORKDIR /app
COPY --from=builder /app /app
RUN mkdir -p /tmp/metrics

ENV PATH="/app/.venv/bin:$PATH"
ENV PROMETHEUS_MULTIPROC_DIR="/tmp/metrics"

EXPOSE 5000

CMD ["python", "entrypoint.py"]
