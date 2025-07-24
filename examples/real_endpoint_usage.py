#!/usr/bin/env python3
"""
Real endpoint usage example for the GPU Worker Pool Client.

This example connects to an actual GPU statistics server running on localhost:8000
and demonstrates real GPU assignment and release operations.

Prerequisites:
1. Start the GPU statistics server:
   python -m gpu_worker_pool.gpu_server
   
2. Verify server is running:
   curl http://localhost:8000/health

The example will:
- Connect to the real GPU server
- Check pool status and available GPUs  
- Request a GPU assignment
- Show GPU utilization and assignment details
- Release the GPU when done

Configuration:
- Memory threshold: 90% (allows high memory usage)
- Utilization threshold: 98% (allows high GPU utilization)
- Demonstrates real-world multi-endpoint client usage

Note: Run this example with PYTHONPATH=. to use the local code:
    PYTHONPATH=. python examples/real_endpoint_usage.py
    
Or install the package in development mode:
    pip install -e .
"""

import os
import sys
import asyncio
import logging

# Add current directory to Python path to use local development code
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["GPU_SERVICE_ENDPOINTS"] = "http://localhost:8000"
os.environ["GPU_MEMORY_THRESHOLD_PERCENT"] = "90.0"
os.environ["GPU_UTILIZATION_THRESHOLD_PERCENT"] = "98.0"  # Allow high utilization for demo
os.environ["GPU_POLLING_INTERVAL"] = "3"

from gpu_worker_pool.client import GPUWorkerPoolClient

# Verify we have the multi-endpoint version
def check_multi_endpoint_support():
    """Check if the current version supports multi-endpoint features."""
    try:
        from gpu_worker_pool.load_balancer import MultiEndpointLoadBalancer
        if not hasattr(MultiEndpointLoadBalancer, 'get_strategy_name'):
            print("ERROR: Using outdated package version!")
            print("Please run with: PYTHONPATH=. python examples/real_endpoint_usage.py")
            print("Or install in development mode: pip install -e .")
            exit(1)
    except ImportError as e:
        print(f"ERROR: Cannot import multi-endpoint components: {e}")
        exit(1)

check_multi_endpoint_support()

# Enable info logging (set to DEBUG for detailed troubleshooting)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

async def main():
    # Multi-endpoint configuration - using same endpoint 3 times to demonstrate load balancing
    client = GPUWorkerPoolClient(
        service_endpoints="http://localhost:8000",  # Single endpoint for real demo
        load_balancing_strategy="availability"  # Choose best available endpoint
    )
    
    async with client:
        # Wait a bit for the client to initialize and get GPU stats
        print("Waiting for client to initialize...")
        await asyncio.sleep(2.0)
        
        # Check pool status first
        status = client.get_pool_status()
        print(f"Pool status: {status.total_gpus} total GPUs, {status.available_gpus} available")
        
        # Get detailed metrics
        metrics = client.get_detailed_metrics()
        print(f"Detailed metrics: {metrics}")
        
        # Request GPU from the best available endpoint
        print("Requesting GPU...")
        assignment = await client.request_gpu(timeout=30.0)
        if assignment:
            # In multi-endpoint mode, check if we have global GPU info
            if hasattr(assignment, 'global_gpu_id') and assignment.global_gpu_id:
                print(f"Assigned GPU {assignment.global_gpu_id} (local: {assignment.gpu_id}) from endpoint {assignment.endpoint_id}")
            else:
                print(f"Assigned GPU {assignment.gpu_id}")
            
            # GPU is automatically released when done
            await client.release_gpu(assignment)
            print("GPU released successfully")
        else:
            print("No GPU assignment received!")

asyncio.run(main())