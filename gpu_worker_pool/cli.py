"""Command-line interface for GPU Worker Pool."""

import asyncio
import json
import sys
from typing import Optional

from .client import GPUWorkerPoolClient


async def status_command():
    """Get the current status of the GPU worker pool."""
    try:
        async with GPUWorkerPoolClient() as client:
            # Wait a moment for initial stats
            await asyncio.sleep(0.5)
            
            # Get pool status
            status = client.get_pool_status()
            
            print("GPU Worker Pool Status:")
            print(f"  Total GPUs: {status.total_gpus}")
            print(f"  Available GPUs: {status.available_gpus}")
            print(f"  Active Workers: {status.active_workers}")
            print(f"  Blocked Workers: {status.blocked_workers}")
            
            if status.total_gpus > 0:
                utilization = (status.total_gpus - status.available_gpus) / status.total_gpus
                print(f"  Utilization: {utilization:.1%}")
            
            # Get detailed metrics
            metrics = client.get_detailed_metrics()
            
            print("\nGPU Details:")
            for gpu in metrics.get('gpu_metrics', []):
                print(f"  GPU {gpu['gpu_id']} ({gpu['name']}):")
                print(f"    Memory: {gpu['memory_usage_percent']:.1f}%")
                print(f"    Utilization: {gpu['utilization_percent']:.1f}%")
                print(f"    Available: {'Yes' if gpu['is_available'] else 'No'}")
                print(f"    Assigned Workers: {gpu['assigned_workers']}")
            
            print(f"\nThresholds:")
            thresholds = metrics.get('thresholds', {})
            print(f"  Memory: {thresholds.get('memory_threshold_percent', 'N/A')}%")
            print(f"  Utilization: {thresholds.get('utilization_threshold_percent', 'N/A')}%")
            
    except Exception as e:
        print(f"Error getting GPU worker pool status: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    """Main CLI entry point."""
    asyncio.run(status_command())


if __name__ == "__main__":
    main()