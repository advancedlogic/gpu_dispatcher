import os
import asyncio

from gpu_worker_pool.client import GPUWorkerPoolClient

os.environ["GPU_SERVICE_ENDPOINT"] = "http://localhost:8000"
os.environ["GPU_MEMORY_THRESHOLD_PERCENT"] = "80.0"
os.environ["GPU_UTILIZATION_THRESHOLD_PERCENT"] = "80.0"
os.environ["GPU_POLLING_INTERVAL"] = "3"

async def main():
    async with GPUWorkerPoolClient() as client:
        # Request a GPU
        assignment = await client.request_gpu()
        print(f"Got GPU {assignment.gpu_id}")
        
        # Do your GPU work here
        await asyncio.sleep(2.0)  # Simulate work
        
        # Release the GPU
        await client.release_gpu(assignment)
        print("GPU released")


if __name__ == "__main__":
    asyncio.run(main())