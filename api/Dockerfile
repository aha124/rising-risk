# Scoring API. Build from the repo root:
#   docker build -f api/Dockerfile -t rising-risk-api .
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# LightGBM needs the OpenMP runtime, which the slim image leaves out.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first, so editing the API code does not reinstall them. The base install
# has no training or warehouse packages. constraints.txt pins the versions CI tests with.
COPY pyproject.toml constraints.txt LICENSE ./
COPY src ./src
RUN pip install -c constraints.txt .

COPY api ./api
COPY artifacts/model ./artifacts/model
COPY artifacts/reports/metrics.json ./artifacts/reports/metrics.json

# The audit log is the only thing the service writes. Mount a volume at /data/audit to keep it.
RUN useradd --system --uid 10001 --no-create-home app \
    && mkdir -p /data/audit \
    && chown app /data/audit

ENV RISKMODEL_MODEL_DIR=/app/artifacts/model \
    RISKMODEL_REPORTS_DIR=/app/artifacts/reports \
    RISKMODEL_AUDIT_DB=/data/audit/audit.db

USER app
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=15s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=2)"]

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
