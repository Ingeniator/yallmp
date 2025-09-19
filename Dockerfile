FROM registry.sberdevices.ru/registry/tools/python:3.11.4

WORKDIR /app

COPY requirements.txt .

COPY pyproject.toml uv.lock ./
# Актуализация зависимостей
RUN pip3 install uv --index-url https://nexus.sberdevices.ru/repository/Pypi/simple/ && \
    python3 -m uv venv .venv  && \
    python3 -m uv sync

COPY . .

# Объявляем порты
EXPOSE 5000

ENV PATH="/app/.venv/bin:${PATH}"

ENTRYPOINT ["./start.sh"]