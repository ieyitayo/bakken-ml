# ─────────────────────────────────────────────────────────────
# Dockerfile — Bakken Basin ML Project
#
# Build:
#   docker build -t bakken-ml .
#
# Train (mount data + persist MLflow runs to the host):
#   docker run --rm \
#     -v "$(pwd)/data:/app/data" \
#     -v "$(pwd)/mlruns:/app/mlruns" \
#     bakken-ml \
#     python src/train.py --config configs/config.yaml
#
# Run the advisor (interactive, needs API key):
#   docker run --rm -it \
#     -v "$(pwd)/data:/app/data" \
#     -v "$(pwd)/mlruns:/app/mlruns" \
#     --env-file .env \
#     bakken-ml \
#     python src/app.py --run-id <best_run_id>
# ─────────────────────────────────────────────────────────────

FROM python:3.11-slim

# Avoid interactive prompts and reduce image size
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps needed by xgboost / matplotlib at runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY . .

# Data, MLflow runs, and configs are expected to be mounted as volumes
# at runtime (see usage examples above) rather than baked into the image.

# Default command runs the test suite so `docker run bakken-ml` verifies
# the build is healthy out of the box. Override with train.py / app.py as needed.
CMD ["pytest", "tests/", "-v"]
