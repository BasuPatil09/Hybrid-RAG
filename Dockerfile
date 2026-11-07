# syntax=docker/dockerfile:1
FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.4,<3.0"

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY pyproject.toml .
COPY src/ src/

RUN pip install --no-deps .

FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    HF_HUB_DISABLE_TELEMETRY=1 \
    GOOGLE_GENAI_HTTP_CLIENT=httpx

RUN groupadd --system --gid 10001 app && \
    useradd --system --uid 10001 --gid app --create-home app && \
    mkdir -p /app/data /home/app/.cache/huggingface && \
    chown -R app:app /app/data /home/app

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY --chown=app:app static/ static/

USER app

EXPOSE 8000

CMD ["uvicorn", "rag.api.main:app", "--host", "0.0.0.0", "--port", "8000"]