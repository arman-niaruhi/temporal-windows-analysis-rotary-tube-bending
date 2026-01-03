import os
import torch
import numpy as np
import random

def enforce_reproducibility(seed: int = 42):
    """
    Configure Python, NumPy, and PyTorch to produce reproducible results.

    This function sets random seeds and forces deterministic
    behavior in CUDA operations where possible.

    Parameters
    ----------
    seed : int, optional
        Seed value for random number generators (default is 42).
    """

    # ============================================================
    # Environment-level seeds
    # ============================================================
    # Ensure consistent hashing for Python objects
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Ensure deterministic behavior in cuBLAS kernels for GPU
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    # ============================================================
    # Seed Python and NumPy RNGs
    # ============================================================
    random.seed(seed)
    np.random.seed(seed)

    # ============================================================
    # Seed PyTorch CPU RNG
    # ============================================================
    torch.manual_seed(seed)

    # ============================================================
    # Seed all available CUDA devices (GPU)
    # ============================================================
    try:
        torch.cuda.manual_seed_all(seed)
    except Exception:
        # If no GPU is available, silently ignore
        pass

    # ============================================================
    # Force deterministic CuDNN behavior
    # ============================================================
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # Disable auto-tuner for determinism

    # ============================================================
    # Use PyTorch deterministic algorithms globally
    # ============================================================
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        # Some PyTorch versions may not support this
        pass
