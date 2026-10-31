# Docker image for the Zurich footfall feature pipeline.
# It installs the project dependencies, copies the source code and configuration,
# and runs the live feature pipeline by default when the container starts.
FROM python:3.10-slim

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock ./

RUN uv sync --locked --no-dev

COPY src ./src
COPY config ./config

ENV PYTHONPATH=/app/src

CMD ["uv", "run", "--no-sync", "python", "-m", "footfall.features.pipeline", "live"]