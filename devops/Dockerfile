FROM baseimages/python311:1.5

ARG GIT_REPO \
    BRANCH_NAME \
    GIT_COMMIT_HASH \
    VERSION_NUMBER

ARG PYPI_INDEX \
    SBEROSC_ENCODED_TOKEN \
    NEXUS_TOKEN_NAME \
    NEXUS_TOKEN_PASS \
    PIPE_PARAM_SERVICE_NAME

ENV UV_INDEX_NEXUS_CI_USERNAME=${NEXUS_TOKEN_NAME} \
    UV_INDEX_NEXUS_CI_PASSWORD=${NEXUS_TOKEN_PASS} \
    UV_INDEX_SBEROSC_USERNAME=token \
    UV_INDEX_SBEROSC_PASSWORD=${SBEROSC_ENCODED_TOKEN}

WORKDIR /usr/src/${PIPE_PARAM_SERVICE_NAME}

COPY pyproject.toml uv.lock ./
# Актуализация зависимостей
RUN pip3 install uv ${PYPI_INDEX} && \
    python3 -m uv venv .venv  && \
    python3 -m uv sync

COPY . .

# Объявляем порты
EXPOSE 5000

ENV PATH="/usr/src/${PIPE_PARAM_SERVICE_NAME}/.venv/bin:${PATH}"

ENTRYPOINT ["./start.sh"]
