# syntax=docker/dockerfile:1.7

############################
# Build stage (optional if you use wheels/poetry etc.)
############################
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Install system deps including browser dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
      curl \
      # Browser dependencies
      libglib2.0-0 \
      libnss3 \
      libnspr4 \
      libdbus-1-3 \
      libatk1.0-0 \
      libatk-bridge2.0-0 \
      libcups2 \
      libdrm2 \
      libexpat1 \
      libxcb1 \
      libxkbcommon0 \
      libx11-6 \
      libxcomposite1 \
      libxdamage1 \
      libxext6 \
      libxfixes3 \
      libxrandr2 \
      libgbm1 \
      libpango-1.0-0 \
      libcairo2 \
      libasound2 \
      libatspi2.0-0 \
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

# Install Playwright browsers for the appuser
RUN /venv/bin/playwright install chromium

# Expose port
EXPOSE 8000

# Healthcheck (hit /health)
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s CMD curl -fsS http://127.0.0.1:8000/health || exit 1

# Run with Gunicorn + Uvicorn worker for automatic memory management
# --max-requests 100: Restart worker after 100 requests to free accumulated memory
# --max-requests-jitter 20: Add randomness (80-120 requests) to prevent simultaneous restarts
# --timeout 120: Allow 2 minutes for long-running Playwright/OpenAI operations
# --workers 1: Single worker to share one browser instance (use semaphore for concurrency)
ENV LOG_LEVEL=INFO
CMD ["/venv/bin/gunicorn", "app.main:app", \
     "--bind", "0.0.0.0:8000", \
     "--workers", "1", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--max-requests", "100", \
     "--max-requests-jitter", "20", \
     "--timeout", "120", \
     "--access-logfile", "-", \
     "--error-logfile", "-"]