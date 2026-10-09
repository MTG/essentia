FROM node:22.18.0-alpine AS frontend
WORKDIR /ui
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM python:3.11.6-slim-bookworm
ENV PYTHONUNBUFFERED=1 PYTHONUTF8=1 QUIETTUNE_DATA=/app/data QUIETTUNE_MODELS=/app/models
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libsndfile1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY backend/requirements-linux.lock backend/requirements-linux.lock
RUN pip install --no-cache-dir -r backend/requirements-linux.lock
COPY backend/ backend/
COPY --from=frontend /ui/dist frontend/dist
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
