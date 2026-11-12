# ---- Build stage: install dependencies into a virtualenv -------------------
# Multi-stage keeps the final image small: build tools never ship to prod.
FROM python:3.13-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# gcc is needed only if a dependency ships source-only wheels; slim images
# lack it. Install, compile, then leave the toolchain behind.
RUN apt-get update && apt-get install -y --no-install-recommends gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /install
COPY requirements.txt .
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip \
    && /opt/venv/bin/pip install -r requirements.txt

# ---- Runtime stage: minimal image with just the app + venv ------------------
FROM python:3.13-slim

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Non-root user: containers shouldn't run as root (defense in depth).
RUN useradd --create-home --shell /bin/bash routeruser

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY app/ ./app/
COPY router.py ./

USER routeruser

EXPOSE 8000

# Liveness probe: fail the container if the API stops answering.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"

# exec-form CMD so PID 1 is uvicorn and SIGTERM (docker stop) reaches it.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers"]
