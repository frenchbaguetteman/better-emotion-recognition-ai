#!/usr/bin/env python3
"""
GPU Monitoring and Troubleshooting Script
Run this during training to diagnose GPU utilization issues
"""

import torch
import subprocess
import time
import os
import argparse

def monitor_gpu():
    """Monitor GPU metrics in real-time"""
    print("\n" + "=" * 80)
    print("GPU MONITORING (Press Ctrl+C to exit)")
    print("=" * 80 + "\n")
    
    try:
        while True:
            os.system('clear' if os.name != 'nt' else 'cls')
            
            print("=" * 80)
            print(f"GPU Status - {time.strftime('%H:%M:%S')}")
            print("=" * 80)
            
            # Run nvidia-smi
            try:
                result = subprocess.run(
                    ['nvidia-smi', '--query-gpu=index,name,memory.used,memory.total,utilization.gpu,utilization.memory,temperature.gpu', 
                     '--format=csv,noheader'],
                    capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0:
                    lines = result.stdout.strip().split('\n')
                    print(f"{'GPU':<4} {'Name':<30} {'Memory Used':<12} {'GPU %':<8} {'Mem %':<8} {'Temp':<6}")
                    print("-" * 80)
                    for line in lines:
                        parts = line.split(',')
                        if len(parts) >= 7:
                            gpu_id = parts[0].strip()
                            gpu_name = parts[1].strip()[:28]
                            mem_used = parts[2].strip()
                            mem_total = parts[3].strip()
                            gpu_util = parts[4].strip()
                            mem_util = parts[5].strip()
                            temp = parts[6].strip()
                            print(f"{gpu_id:<4} {gpu_name:<30} {mem_used:>8}/{mem_total:<8} {gpu_util:>7} {mem_util:>7} {temp:>5}")
            except Exception as e:
                print(f"Error running nvidia-smi: {e}")
                print("Is nvidia-smi installed?")
            
            print("\n" + "=" * 80)
            print("KEY METRICS:")
            print("=" * 80)
            
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    with torch.cuda.device(i):
                        allocated = torch.cuda.memory_allocated() / 1e9
                        reserved = torch.cuda.memory_reserved() / 1e9
                        total = torch.cuda.get_device_properties(i).total_memory / 1e9
                        
                        print(f"GPU {i}:")
                        print(f"  Allocated: {allocated:.2f} GB ({allocated/total*100:.1f}%)")
                        print(f"  Reserved:  {reserved:.2f} GB ({reserved/total*100:.1f}%)")
                        print(f"  Total:     {total:.2f} GB")
                        
                        if allocated > total * 0.9:
                            print("  ⚠️  WARNING: GPU memory nearly full!")
                        elif allocated < total * 0.3:
                            print("  ⚠️  WARNING: GPU memory underutilized!")
                        else:
                            print("  ✓ GPU memory usage OK")
            
            print("\n" + "=" * 80)
            print("RECOMMENDATIONS IF GPU UNDERUTILIZED:")
            print("=" * 80)
            print("""
1. Increase batch size (try 24, 32, 48)
2. Reduce num_workers to 1-2
3. Enable prefetch_factor in DataLoader
4. Use gradient accumulation wisely
5. Check if CPU is bottlenecking (monitor CPU usage)
6. Enable mixed precision training
7. Check for CPU to GPU data transfer bottlenecks

GPU Utilization should be:
  ✓ 80-95%+ during training
  ✓ Stable (not fluctuating)
  ✓ Memory should stay relatively constant
            """)
            
            time.sleep(2)
    
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")

def diagnose():
    """Run diagnostics on GPU setup"""
    print("\n" + "=" * 80)
    print("GPU DIAGNOSTICS")
    print("=" * 80 + "\n")
    
    # Check CUDA
    print("1. CUDA Availability:")
    print(f"   CUDA Available: {torch.cuda.is_available()}")
    print(f"   CUDA Version: {torch.version.cuda}")
    print(f"   cuDNN Version: {torch.backends.cudnn.version()}")
    
    if torch.cuda.is_available():
        print(f"   Device Count: {torch.cuda.device_count()}")
        
        # Check each GPU
        print("\n2. GPU Details:")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"   GPU {i}: {props.name}")
            print(f"      Compute Capability: {props.major}.{props.minor}")
            print(f"      Total Memory: {props.total_memory / 1e9:.2f} GB")
            print(f"      Max Threads Per Block: {props.max_threads_per_block}")
            print(f"      Max Threads Per SM: {props.max_threads_per_multiprocessor}")
            print(f"      Multi Processor Count: {props.multi_processor_count}")
        
        # Check if we can do operations
        print("\n3. GPU Operations Test:")
        try:
            x = torch.randn(1000, 1000).cuda()
            y = torch.randn(1000, 1000).cuda()
            z = torch.matmul(x, y)
            print("   ✓ Basic CUDA operations work")
        except Exception as e:
            print(f"   ✗ CUDA operations failed: {e}")
        
        # Check TF32
        print("\n4. Optimization Flags:")
        print(f"   TF32 (matmul): {torch.backends.cuda.matmul.allow_tf32}")
        print(f"   TF32 (cudnn): {torch.backends.cudnn.allow_tf32}")
        print(f"   cuDNN Benchmark: {torch.backends.cudnn.benchmark}")
        print(f"   cuDNN Deterministic: {torch.backends.cudnn.deterministic}")
    else:
        print("   ✗ CUDA not available!")
    
    # Check environment variables
    print("\n5. Environment Variables:")
    cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')
    cuda_device_order = os.environ.get('CUDA_DEVICE_ORDER', 'Not set')
    cublas_config = os.environ.get('CUBLAS_WORKSPACE_CONFIG', 'Not set')
    
    print(f"   CUDA_VISIBLE_DEVICES: {cuda_visible_devices}")
    print(f"   CUDA_DEVICE_ORDER: {cuda_device_order}")
    print(f"   CUBLAS_WORKSPACE_CONFIG: {cublas_config}")
    
    print("\n" + "=" * 80)

def main():
    parser = argparse.ArgumentParser(description="GPU Monitoring and Diagnostics")
    parser.add_argument('--monitor', action='store_true', help='Monitor GPU in real-time')
    parser.add_argument('--diagnose', action='store_true', help='Run GPU diagnostics')
    
    args = parser.parse_args()
    
    if args.monitor:
        monitor_gpu()
    elif args.diagnose:
        diagnose()
    else:
        # Default: run diagnostics
        diagnose()

if __name__ == "__main__":
    main()
