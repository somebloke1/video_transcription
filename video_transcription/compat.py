"""
Compatibility patches for various library version conflicts.

This module provides patches for issues arising from version mismatches
between libraries like NeMo and transformers.
"""

_patch_applied = False


def apply_nemo_transformers_patch() -> None:
    """
    Apply compatibility patch for NeMo/transformers version conflict.

    NeMo expects PytorchGELUTanh which was removed in transformers 4.50+.
    This function creates a compatibility shim if the class is missing.

    This patch is idempotent - calling it multiple times is safe.
    """
    global _patch_applied
    if _patch_applied:
        return

    try:
        from transformers.activations import PytorchGELUTanh  # noqa: F401
        # Class exists, no patch needed
    except ImportError:
        import torch.nn as nn
        from transformers import activations as activations_module

        class PytorchGELUTanh(nn.Module):
            """Compatibility shim for older NeMo versions."""
            def __init__(self):
                super().__init__()
                self.act = nn.GELU(approximate="tanh")

            def forward(self, x):
                return self.act(x)

        # Inject into transformers.activations
        activations_module.PytorchGELUTanh = PytorchGELUTanh
        print("   [Patched PytorchGELUTanh for NeMo compatibility]")

    _patch_applied = True
