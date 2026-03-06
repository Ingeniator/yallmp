### Stage 1: install dependencies
FROM baseimages/python311:1.5 AS builder

ARG PYPI_INDEX

ENV UV_INDEX_NEXUS_USERNAME=${NEXUS_TOKEN_NAME} \
    UV_INDEX_NEXUS_PASSWORD=${NEXUS_TOKEN_PASS}

WORKDIR /usr/src/${PIPE_PARAM_SERVICE_NAME}

COPY pyproject.toml uv.lock ./
RUN pip3 install uv ${PYPI_INDEX} && \
    python3 -m uv venv .venv && \
    python3 -m uv sync

### Stage 2: runtime
FROM baseimages/python311:1.5

ARG GIT_REPO \
    BRANCH_NAME \
    GIT_COMMIT_HASH \
    VERSION_NUMBER \
    PIPE_PARAM_SERVICE_NAME

WORKDIR /usr/src/${PIPE_PARAM_SERVICE_NAME}

COPY --from=builder /usr/src/${PIPE_PARAM_SERVICE_NAME}/.venv .venv
COPY . .

EXPOSE 5000

ENV PATH="/usr/src/${PIPE_PARAM_SERVICE_NAME}/.venv/bin:${PATH}"

ENTRYPOINT ["./start.sh"]
