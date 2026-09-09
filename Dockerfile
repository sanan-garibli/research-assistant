# Async Research Assistant (Topic 4) — CLI-driven project, no HTTP server.
FROM python:3.12-slim

LABEL org.opencontainers.image.title="Async Research Assistant (Topic 4)"
LABEL org.opencontainers.image.version="1.0"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install dependencies before copying source so Docker can cache this layer.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Non-root user; also owns the on-disk cache directory it will create at runtime.
RUN useradd --create-home --shell /bin/bash appuser \
    && mkdir -p /app/.cache \
    && chown -R appuser:appuser /app
USER appuser

# Runs all 5 sample research questions end-to-end with one command.
# Override to run a single question: `docker run --env-file .env <image> ask "..."`
# Or the graded smoke tests: `docker run --env-file .env <image> pytest tests/test_ai_smoke.py -v`
CMD ["python", "-m", "researcher", "demo"]
