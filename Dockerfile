# Container for the analysis / plotting stages of the pipeline.
#
# The CPU stages (clustering, geometry, figures, tests) run fully inside this
# image. The GPU capture stages (field dumps, transcoder features, steering
# sweeps) need an NVIDIA host with the Qwen3-4B weights mounted and are run
# outside the container — see README "Usage".
#
# Build:  docker build -t qwen-circuits .
# Test:   docker run --rm qwen-circuits pytest -q
# Shell:  docker run --rm -it -v "$PWD":/app qwen-circuits bash
# Jupyter:docker run --rm -p 8888:8888 qwen-circuits \
#             jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root

FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# minimal build tooling for wheels that need compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git \
    && rm -rf /var/lib/apt/lists/*

# install CPU-only torch first (small), then the rest of the project deps
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml requirements.txt README.md ./
COPY src ./src
RUN pip install -e ".[dev,notebooks]"

# copy the remaining project (scripts, configs, tests, results, docs)
COPY . .

EXPOSE 8888

CMD ["bash"]
