# ---- frontend build (Vite → dist) ----
FROM node:22-bookworm-slim AS frontend
WORKDIR /fe
RUN npm install -g pnpm@10.32.1
COPY frontend/package.json frontend/pnpm-lock.yaml ./
RUN pnpm install --frozen-lockfile
COPY frontend/ ./
# Same-origin: intentionally do NOT set VITE_WS_BACKEND_URL / VITE_HTTP_BACKEND_URL.
RUN pnpm build

# ---- backend runtime (FastAPI serves API + WS + built SPA) ----
FROM python:3.12-slim-bookworm AS runtime
ENV PYTHONUNBUFFERED=1
WORKDIR /app
RUN pip install --no-cache-dir "poetry==2.4.1"
COPY backend/pyproject.toml backend/poetry.lock ./
RUN poetry config virtualenvs.create false \
    && poetry install --only main --no-interaction --no-ansi --no-root
RUN python -m playwright install --with-deps chromium
COPY backend/ ./
COPY --from=frontend /fe/dist ./static
EXPOSE 8080
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}"]
