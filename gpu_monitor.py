#!/usr/bin/env python3
"""
GPU Idle Time Monitor - Measures idle seconds per minute for each GPU.
"""

import subprocess
import time
import sys
from datetime import datetime
from collections import defaultdict

def get_gpu_utilization():
    """Get GPU utilization for all GPUs."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5
        )
        utils = {}
        for line in result.stdout.strip().split('\n'):
            if line:
                parts = line.split(',')
                gpu_id = int(parts[0].strip())
                util = int(parts[1].strip())
                utils[gpu_id] = util
        return utils
    except Exception as e:
        print(f"Error reading GPU stats: {e}")
        return {}

def main():
    sample_interval = 1  # seconds between samples
    report_interval = 60  # seconds between reports
    idle_threshold = 5  # GPU-Util <= this is considered "idle"

    print(f"GPU Idle Monitor - Sampling every {sample_interval}s, reporting every {report_interval}s")
    print(f"Idle threshold: GPU-Util <= {idle_threshold}%")
    print("-" * 80)

    # Track idle samples per GPU
    idle_samples = defaultdict(int)
    total_samples = 0
    last_report = time.time()

    try:
        while True:
            utils = get_gpu_utilization()
            if utils:
                total_samples += 1
                for gpu_id, util in utils.items():
                    if util <= idle_threshold:
                        idle_samples[gpu_id] += 1

            # Check if it's time to report
            now = time.time()
            if now - last_report >= report_interval:
                if total_samples > 0:
                    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Last {report_interval}s report ({total_samples} samples):")
                    print("-" * 60)

                    total_idle_seconds = 0
                    gpu_ids = sorted(utils.keys()) if utils else sorted(idle_samples.keys())

                    for gpu_id in gpu_ids:
                        idle_count = idle_samples[gpu_id]
                        idle_seconds = idle_count * sample_interval
                        idle_pct = (idle_count / total_samples) * 100
                        total_idle_seconds += idle_seconds

                        bar_len = int(idle_pct / 5)  # 20 chars max
                        bar = "█" * bar_len + "░" * (20 - bar_len)

                        print(f"  GPU {gpu_id}: {idle_seconds:3.0f}s idle ({idle_pct:5.1f}%) [{bar}]")

                    avg_idle = total_idle_seconds / len(gpu_ids) if gpu_ids else 0
                    total_gpu_seconds = report_interval * len(gpu_ids)
                    efficiency = ((total_gpu_seconds - total_idle_seconds) / total_gpu_seconds) * 100 if total_gpu_seconds > 0 else 0

                    print("-" * 60)
                    print(f"  TOTAL: {total_idle_seconds:.0f}s idle across all GPUs")
                    print(f"  AVG per GPU: {avg_idle:.1f}s idle per minute")
                    print(f"  EFFICIENCY: {efficiency:.1f}% (time GPUs were working)")
                    print("-" * 60)

                # Reset counters
                idle_samples = defaultdict(int)
                total_samples = 0
                last_report = now

            time.sleep(sample_interval)

    except KeyboardInterrupt:
        print("\nMonitor stopped.")
        sys.exit(0)

if __name__ == "__main__":
    main()
