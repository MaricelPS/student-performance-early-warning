# --- build stage: build a wheel with the runtime dependencies only -----------
FROM python:3.13-slim AS build
WORKDIR /build
COPY pyproject.toml README.md requirements.lock ./
COPY src ./src
RUN pip install --no-cache-dir --upgrade pip wheel \
 && pip wheel --no-cache-dir --wheel-dir /wheels -c requirements.lock .

# --- runtime stage ------------------------------------------------------------
FROM python:3.13-slim
# libgomp is required by LightGBM
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 curl \
 && rm -rf /var/lib/apt/lists/*
RUN useradd --create-home --uid 10001 app
WORKDIR /app
COPY --from=build /wheels /wheels
RUN pip install --no-cache-dir /wheels/* && rm -rf /wheels
COPY models ./models

ENV EW_MODELS_DIR=/app/models \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 \
    WEB_CONCURRENCY=2
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s \
  CMD curl -fsS http://localhost:8000/health || exit 1
CMD ["sh", "-c", "uvicorn early_warning.api.app:app --host 0.0.0.0 --port 8000 --workers ${WEB_CONCURRENCY}"]
