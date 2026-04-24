# Dockerfile for serving a PI policy on TPU.
# Using specific XLA base image as requested.

FROM us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/xla:r2.6.0_3.11_tpuvm_cxx11

COPY --from=ghcr.io/astral-sh/uv:0.5.1 /uv /uvx /bin/

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y git git-lfs curl build-essential clang

# Copy from the cache instead of linking since it's a mounted volume
ENV UV_LINK_MODE=copy

# Write the virtual environment outside of the project directory
ENV UV_PROJECT_ENVIRONMENT=/.venv

# Setup the virtual environment and sync non-TPU dependencies
# Match base image python version (3.10)
RUN uv venv --python 3.10 $UV_PROJECT_ENVIRONMENT
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=packages/openpi-client/pyproject.toml,target=packages/openpi-client/pyproject.toml \
    --mount=type=bind,source=packages/openpi-client/src,target=packages/openpi-client/src \
    GIT_LFS_SKIP_SMUDGE=1 uv sync --frozen --no-install-project --no-dev

# Install JAX with TPU support using standard pip (no version pinning, no libtpu mention)
# RUN /.venv/bin/pip install chex distrax && \
#     /.venv/bin/pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html && \
#     /.venv/bin/pip install --upgrade rich

# Copy transformers_replace files while preserving directory structure
COPY src/openpi/models_pytorch/transformers_replace/ /tmp/transformers_replace/
RUN /.venv/bin/python -c "import transformers; print(transformers.__file__)" | xargs dirname | xargs -I{} cp -r /tmp/transformers_replace/* {} && rm -rf /tmp/transformers_replace

# Manual tokenizer setup
RUN mkdir -p /openpi_assets/big_vision && \
    curl -o /openpi_assets/big_vision/paligemma_tokenizer.model https://storage.googleapis.com/big_vision/paligemma_tokenizer.model

ENV OPENPI_DATA_HOME=/openpi_assets

CMD /bin/bash -c "uv run scripts/serve_policy.py $SERVER_ARGS"
