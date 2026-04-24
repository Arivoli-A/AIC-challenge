import os

import pynvml
import pytest


def set_jax_backend_if_no_accelerator() -> None:
    # Check for GPU
    has_gpu = False
    try:
        pynvml.nvmlInit()
        pynvml.nvmlShutdown()
        has_gpu = True
    except pynvml.NVMLError:
        pass

    # Check for TPU
    has_tpu = False
    if not has_gpu:
        try:
            # TPU check can be done by looking for TPU-related env vars or devices
            # One simple check is looking for the presence of TPU-related library or device
            if os.path.exists("/dev/accel0") or "TPU_NAME" in os.environ:
                has_tpu = True
        except Exception:
            pass

    if not has_gpu and not has_tpu:
        # No GPU or TPU found.
        os.environ["JAX_PLATFORMS"] = "cpu"


def pytest_configure(config: pytest.Config) -> None:
    set_jax_backend_if_no_accelerator()
