FROM ghcr.io/astral-sh/uv:python3.10-bookworm-slim

WORKDIR /app

ENV UV_HTTP_TIMEOUT=300

COPY pyproject.toml uv.lock README.md ./

RUN uv sync --frozen --no-dev --no-install-project

COPY src ./src
COPY configs ./configs
COPY models ./models

RUN uv sync --frozen --no-dev

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app/src"
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]