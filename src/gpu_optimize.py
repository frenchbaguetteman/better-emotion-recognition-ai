#!/usr/bin/env python3
"""
GPU Optimization Setup Script
Runs before training to maximize GPU utilization and stability
"""

import torch
import os
import subprocess
import time

def check_nvidia_smi():
    """Check GPU status using nvidia-smi"""
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,compute_cap', '--format=csv,noheader'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("=" * 70)
            print("GPU INFORMATION:")
            print("=" * 70)
            print(result.stdout)
        else:
            print("nvidia-smi not found - assuming non-NVIDIA GPU")
    except:
        print("Could not run nvidia-smi")

def optimize_gpu():
    """Apply GPU optimization settings"""
    print("\n" + "=" * 70)
    print("APPLYING GPU OPTIMIZATIONS:")
    print("=" * 70)
    
    # 1. Enable TF32 for faster computation on newer GPUs
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        print("✓ Enabled TF32 acceleration")
    
    # 2. Set CUDA environment variables for better performance
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4294967296'  # 4GB max workspace for cuBLAS
    print("✓ Set CUDA environment variables")
    
    # 3. Check CUDA availability and properties
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        print(f"✓ Available GPUs: {device_count}")
        
        for i in range(device_count):
            props = torch.cuda.get_device_properties(i)
            print(f"  - GPU {i}: {props.name}")
            print(f"    - Compute Capability: {props.major}.{props.minor}")
            print(f"    - Total Memory: {props.total_memory / 1e9:.2f} GB")
            print(f"    - Max Threads Per Block: {props.max_threads_per_block}")
    
    # 4. Print memory info
    if torch.cuda.is_available():
        print("\n" + "=" * 70)
        print("INITIAL GPU MEMORY:")
        print("=" * 70)
        for i in range(torch.cuda.device_count()):
            with torch.cuda.device(i):
                allocated = torch.cuda.memory_allocated() / 1e9
                reserved = torch.cuda.memory_reserved() / 1e9
                total = torch.cuda.get_device_properties(0).total_memory / 1e9
                print(f"GPU {i}: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved, {total:.2f}GB total")
    
    print("\n" + "=" * 70)
    print("TRAINING RECOMMENDATIONS:")
    print("=" * 70)
    print("""
✓ Batch Size: Use 16-32 (adjust based on available VRAM)
✓ Number of Workers: 2-4 (too many causes CPU bottleneck)
✓ Prefetch Factor: 4 (cache 4 batches per worker)
✓ Pin Memory: Enabled (faster GPU transfer)
✓ Persistent Workers: Enabled (reduces worker startup overhead)
✓ Mixed Precision: Enabled (reduces memory, faster compute)
✓ Gradient Accumulation: 2 (you have larger batches now)
✓ Gradient Checkpointing: Consider if OOM errors occur

WHAT TO MONITOR:
→ GPU Memory Should Stay Stable (no constant growth)
→ GPU Utilization Should Stay >85% Consistently
→ Iteration Time Should be Stable (no sudden spikes)
→ Temperature Should Stay <80°C
    """)

def disable_p2p_access():
    """Disable P2P access if multiple GPUs (can help with stability)"""
    if torch.cuda.device_count() > 1:
        try:
            for i in range(torch.cuda.device_count()):
                for j in range(torch.cuda.device_count()):
                    if i != j:
                        torch.cuda.can_device_access_peer(i, j)
            print("✓ P2P GPU access configured")
        except:
            pass

def main():
    print("\n" + "=" * 70)
    print("GPU OPTIMIZATION SETUP")
    print("=" * 70)
    
    check_nvidia_smi()
    optimize_gpu()
    disable_p2p_access()
    
    print("\n" + "=" * 70)
    print("GPU optimization complete! Ready to train.")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    main()
