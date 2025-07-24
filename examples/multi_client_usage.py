import os

os.environ["GPU_SERVICE_ENDPOINT"] = "http://localhost:8000"
os.environ["GPU_MEMORY_THRESHOLD_PERCENT"] = "80.0"
os.environ["GPU_UTILIZATION_THRESHOLD_PERCENT"] = "80.0"
os.environ["GPU_POLLING_INTERVAL"] = "3"

import asyncio
from gpu_worker_pool.client import GPUWorkerPoolClient, GPUContextManager


async def worker_task(client, worker_id, duration):
    """Simulate a worker task."""
    try:
        async with GPUContextManager(client, timeout=120.0) as gpu_id:
            print(f"Worker {worker_id} got GPU {gpu_id}")
            await asyncio.sleep(duration)
            print(f"Worker {worker_id} finished")
            return f"worker-{worker_id}-success"
    except Exception as e:
        print(f"Worker {worker_id} failed: {e}")
        return f"worker-{worker_id}-failed"

async def run_multiple_workers():
    async with GPUWorkerPoolClient() as client:
        # Create multiple worker tasks
        tasks = [
            worker_task(client, i, 1.0 + (i * 0.5))
            for i in range(5)
        ]
        
        # Run workers concurrently
        results = await asyncio.gather(*tasks)
        
        # Analyze results
        successes = [r for r in results if r.endswith('-success')]
        failures = [r for r in results if r.endswith('-failed')]
        
        print(f"Completed: {len(successes)} successes, {len(failures)} failures")

asyncio.run(run_multiple_workers())