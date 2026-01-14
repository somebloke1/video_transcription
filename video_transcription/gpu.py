"""
GPU memory management utilities.

This module provides functions for managing GPU memory,
particularly important when loading multiple large models sequentially.
"""

import gc
from typing import Optional

# Lazy import torch to avoid forcing GPU-dependent imports
_torch = None


def _get_torch():
    """Lazy import torch."""
    global _torch
    if _torch is None:
        import torch
        _torch = torch
    return _torch


def cleanup_gpu(aggressive: bool = False) -> None:
    """
    Free GPU memory.

    Runs garbage collection and clears CUDA cache. Use aggressive=True
    for more thorough cleanup when switching between large models.

    Args:
        aggressive: If True, runs multiple gc passes and synchronizes CUDA.
    """
    gc.collect()
    if aggressive:
        gc.collect()  # Run twice for thorough cleanup

    torch = _get_torch()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        if aggressive:
            torch.cuda.synchronize()  # Wait for all operations to complete
            torch.cuda.reset_peak_memory_stats()
            gc.collect()


def get_device() -> str:
    """
    Get the best available device for computation.

    Returns:
        'cuda' if CUDA is available, otherwise 'cpu'.
    """
    torch = _get_torch()
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_available_vram() -> Optional[float]:
    """
    Get available VRAM in GB.

    Returns:
        Available VRAM in GB, or None if CUDA is not available.
    """
    torch = _get_torch()
    if not torch.cuda.is_available():
        return None
    free_mem = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()
    return free_mem / 1e9


def get_gpu_info() -> dict:
    """
    Get GPU information.

    Returns:
        Dictionary with 'name', 'total_vram', and 'available_vram' keys,
        or empty dict if CUDA is not available.
    """
    torch = _get_torch()
    if not torch.cuda.is_available():
        return {}

    props = torch.cuda.get_device_properties(0)
    return {
        "name": torch.cuda.get_device_name(),
        "total_vram": props.total_memory / 1e9,
        "available_vram": get_available_vram()
    }


def print_gpu_status() -> None:
    """Print current GPU status to console."""
    torch = _get_torch()
    if torch.cuda.is_available():
        info = get_gpu_info()
        print(f"🖥️ Device: cuda")
        print(f"🎮 GPU: {info['name']}")
        print(f"💾 VRAM: {info['total_vram']:.1f} GB total, {info['available_vram']:.1f} GB available")
    else:
        print("🖥️ Device: cpu (no CUDA available)")
