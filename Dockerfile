FROM python:3.13-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libxcb1 \
    libgomp1 \
    libgl1 \
    libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

# Install dependencies (cached layer)
COPY pyproject.toml uv.lock* ./
RUN uv sync --frozen --no-install-project

# Copy source
COPY src/ ./src/
COPY alembic/ ./alembic/
COPY alembic.ini ./
COPY README.md ./
COPY data/raw/ ./data/raw/

# Sync again to install the project itself
RUN uv sync --frozen
