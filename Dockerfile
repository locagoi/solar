# syntax=docker/dockerfile:1.7

############################
# Build stage (optional if you use wheels/poetry etc.)
############################
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Install system deps (if you need psycopg or similar, add build deps here)
RUN apt-get update && apt-get install -y --no-install-recommends \
      curl \
    && rm -rf /var/lib/apt/lists/*

# Create user
RUN useradd -m -u 10001 appuser

WORKDIR /app

# Copy dependency list first for better caching
COPY requirements.txt .

# Use a venv to keep things tidy (optional)
RUN python -m venv /venv && /venv/bin/pip install --upgrade pip \
    && /venv/bin/pip install -r requirements.txt

# Copy app code
COPY app ./app

# Switch to non-root
USER appuser

# Expose port
EXPOSE 8000

# Healthcheck (hit /health)
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s CMD curl -fsS http://127.0.0.1:8000/health || exit 1

# Run uvicorn directly
CMD ["/venv/bin/uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]