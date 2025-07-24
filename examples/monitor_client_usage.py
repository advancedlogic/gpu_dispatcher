import os

os.environ["GPU_SERVICE_ENDPOINT"] = "http://localhost:8000"
os.environ["GPU_MEMORY_THRESHOLD_PERCENT"] = "80.0"
os.environ["GPU_UTILIZATION_THRESHOLD_PERCENT"] = "80.0"
os.environ["GPU_POLLING_INTERVAL"] = "3"

import asyncio
from gpu_worker_pool.client import GPUWorkerPoolClient

async def monitor_and_scale():
    async with GPUWorkerPoolClient() as client:
        while True:
            status = client.get_pool_status()
            metrics = client.get_detailed_metrics()
            
            # Calculate utilization
            if status.total_gpus > 0:
                utilization = (status.total_gpus - status.available_gpus) / status.total_gpus
                
                print(f"Pool Status:")
                print(f"  Total GPUs: {status.total_gpus}")
                print(f"  Available GPUs: {status.available_gpus}")
                print(f"  Utilization: {utilization:.1%}")
                print(f"  Active workers: {status.active_workers}")
                print(f"  Blocked workers: {status.blocked_workers}")
                
                # Scaling decisions
                if utilization > 0.9 and status.blocked_workers > 0:
                    print("⚠️  High utilization with blocked workers - consider scaling")
                elif utilization < 0.3:
                    print("ℹ️  Low utilization - consider scaling down")
                else:
                    print("✅ Utilization is optimal")
            
            await asyncio.sleep(10)  # Monitor every 10 seconds

# Run monitoring (would typically run in background)
asyncio.run(monitor_and_scale())