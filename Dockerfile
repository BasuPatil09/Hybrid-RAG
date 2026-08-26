FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd --system --gid 10001 app && \
    useradd --system --uid 10001 --gid app --create-home app && \
    mkdir -p /opt/venv /app/data /home/app/.cache/huggingface && \
    chown -R app:app /opt/venv /app /home/app

USER app

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY --chown=app:app requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY --chown=app:app pyproject.toml .
COPY --chown=app:app src/ src/
RUN pip install --no-cache-dir --no-deps .

COPY --chown=app:app static/ static/

EXPOSE 8000

CMD ["uvicorn", "rag.api.main:app", "--host", "0.0.0.0", "--port", "8000"]