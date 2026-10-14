# ReVive container: React UI built with Node, served by FastAPI + the OpenCV 5 agent.
# Built for ARM64 so it runs natively on AWS Graviton:
#   docker build --platform linux/arm64 -t revive .
#   docker run -p 8000:8000 revive

# ---- stage 1: build the web UI ----
FROM node:22-slim AS web
WORKDIR /web
COPY web/package.json web/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY web/ ./
RUN npm run build

# ---- stage 2: Python runtime ----
FROM python:3.14-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    REVIVE_DATA_DIR=/tmp/revive-jobs
WORKDIR /app

# scikit-image is only needed by the offline evaluation, not the web app
COPY requirements.txt ./
RUN grep -v '^scikit-image' requirements.txt > requirements-runtime.txt \
 && pip install --no-cache-dir -r requirements-runtime.txt

COPY revive/ revive/
COPY server/ server/
COPY --from=web /web/dist web/dist

RUN useradd --create-home --uid 10001 revive
USER revive

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=2)"

# One process on purpose: jobs and their live event streams live in memory.
# Restorations run on threads (OpenCV releases the GIL), REVIVE_WORKERS at a time.
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers", "--forwarded-allow-ips", "*"]
