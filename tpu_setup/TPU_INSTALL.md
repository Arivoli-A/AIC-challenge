# TPU Installation Guide for openpi

To set up `openpi` on a Google Cloud TPU VM, follow these steps:

### 1. Install `uv`
If `uv` is not already installed:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
```

### 2. Sync Project Environment
Use the high-level `sync` command with the `tpu` extra. This will automatically resolve and install all dependencies (including JAX for TPU and transitive dependencies like `chex` and `distrax`) into a managed virtual environment:
```bash
uv sync --extra tpu
```

### 3. Verify TPU Detection
Run this command within the managed environment to confirm JAX sees the TPU:
```bash
uv run python3 -c 'import jax; print(f"Devices: {jax.devices()}"); print(f"Backend: {jax.extend.backend.get_backend().platform}")'
```
The output should list `TpuDevice` and show `Backend: tpu`.
