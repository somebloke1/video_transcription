#!/usr/bin/env python3
"""
Clear orphaned GPU memory from CUDA.
Run this if nvidia-smi shows memory usage but no processes.
"""

import gc
import torch

print("🧹 Clearing GPU memory...")

# Force garbage collection
gc.collect()

# Clear CUDA cache for all available devices
if torch.cuda.is_available():
    num_gpus = torch.cuda.device_count()
    print(f"   Found {num_gpus} GPU(s)")

    for i in range(num_gpus):
        with torch.cuda.device(i):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            print(f"   ✓ Cleared GPU {i}")

    print("✅ GPU memory cleared!")
else:
    print("⚠️  No CUDA devices found")

# Print current memory usage
if torch.cuda.is_available():
    print("\n📊 Current GPU Memory Status:")
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / 1024**3
        reserved = torch.cuda.memory_reserved(i) / 1024**3
        print(f"   GPU {i}: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")
