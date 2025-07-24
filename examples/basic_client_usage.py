#!/usr/bin/env python3
"""
Basic usage example for the GPU Worker Pool Client.

This example demonstrates how to use the GPUWorkerPoolClient to request
and release GPU resources in a simple application.

Note: Run this example with PYTHONPATH=. to use the local code:
    PYTHONPATH=. python examples/basic_client_usage.py
    
Or install the package in development mode:
    pip install -e .
"""

import asyncio
import logging
import inspect
from datetime import datetime
from unittest.mock import AsyncMock, patch

from gpu_worker_pool.client import GPUWorkerPoolClient, gpu_worker_pool_client, GPUContextManager

# Check for multi-endpoint support (optional for basic usage)
def check_version():
    """Check if we have the latest version with multi-endpoint support."""
    sig = inspect.signature(GPUWorkerPoolClient.__init__)
    if 'service_endpoints' not in sig.parameters:
        print("Note: Running with older version (multi-endpoint features not available)")
        print("For full multi-endpoint support, run: PYTHONPATH=. python examples/basic_client_usage.py")
        return False
    return True

has_multi_endpoint = check_version()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


async def example_basic_usage():
    """Example of basic client usage with manual start/stop."""
    print("=== Basic Client Usage Example ===")
    
    # Mock the HTTP client for this example
    with patch('gpu_worker_pool.http_client.aiohttp.ClientSession') as mock_session_class:
        # Set up mock HTTP responses
        mock_session = AsyncMock()
        mock_response = AsyncMock()
        mock_response.json.return_value = {
            "gpu_count": 2,
            "total_memory_mb": 16384,
            "total_used_memory_mb": 4096,
            "average_utilization_percent": 25.0,
            "gpus_summary": [
                {"gpu_id": 0, "name": "GPU-0", "memory_usage_percent": 20.0, "utilization_percent": 15.0},
                {"gpu_id": 1, "name": "GPU-1", "memory_usage_percent": 30.0, "utilization_percent": 35.0}
            ],
            "total_memory_usage_percent": 25.0,
            "timestamp": datetime.now().isoformat()
        }
        mock_response.status = 200
        mock_session.get.return_value.__aenter__.return_value = mock_response
        mock_session_class.return_value = mock_session
        
        # Create and start client
        client = GPUWorkerPoolClient(
            memory_threshold=80.0,
            utilization_threshold=90.0,
            worker_timeout=30.0
        )
        
        try:
            print("Starting GPU worker pool client...")
            await client.start()
            print("Client started successfully!")
            
            # Wait a moment for GPU monitor to get initial stats
            await asyncio.sleep(0.2)
            
            # Get pool status
            status = client.get_pool_status()
            print(f"Pool status: {status.total_gpus} total GPUs, {status.available_gpus} available")
            
            # Request a GPU
            print("Requesting GPU assignment...")
            assignment = await client.request_gpu(timeout=10.0)
            print(f"GPU {assignment.gpu_id} assigned to worker {assignment.worker_id}")
            
            # Simulate some work
            print("Simulating work on GPU...")
            await asyncio.sleep(1.0)
            
            # Check status while GPU is assigned
            status = client.get_pool_status()
            print(f"Pool status during work: {status.active_workers} active workers, {status.available_gpus} available GPUs")
            
            # Release the GPU
            print("Releasing GPU...")
            await client.release_gpu(assignment)
            print("GPU released successfully!")
            
            # Final status check
            status = client.get_pool_status()
            print(f"Final pool status: {status.active_workers} active workers, {status.available_gpus} available GPUs")
            
        finally:
            print("Stopping client...")
            await client.stop()
            print("Client stopped.")


async def example_context_manager_usage():
    """Example using the client as a context manager."""
    print("\n=== Context Manager Usage Example ===")
    
    # Mock the HTTP client for this example
    with patch('gpu_worker_pool.http_client.aiohttp.ClientSession') as mock_session_class:
        # Set up mock HTTP responses
        mock_session = AsyncMock()
        mock_response = AsyncMock()
        mock_response.json.return_value = {
            "gpu_count": 1,
            "total_memory_mb": 8192,
            "total_used_memory_mb": 2048,
            "average_utilization_percent": 25.0,
            "gpus_summary": [
                {"gpu_id": 0, "name": "Test-GPU", "memory_usage_percent": 25.0, "utilization_percent": 30.0}
            ],
            "total_memory_usage_percent": 25.0,
            "timestamp": datetime.now().isoformat()
        }
        mock_response.status = 200
        mock_session.get.return_value.__aenter__.return_value = mock_response
        mock_session_class.return_value = mock_session
        
        # Use client as context manager
        async with GPUWorkerPoolClient() as client:
            print("Client started automatically via context manager")
            
            # Wait for initial stats
            await asyncio.sleep(0.2)
            
            # Request and release GPU
            assignment = await client.request_gpu()
            print(f"GPU {assignment.gpu_id} assigned")
            
            await asyncio.sleep(0.5)  # Simulate work
            
            await client.release_gpu(assignment)
            print("GPU released")
        
        print("Client stopped automatically via context manager")


async def example_factory_function():
    """Example using the factory function."""
    print("\n=== Factory Function Usage Example ===")
    
    # Mock the HTTP client for this example
    with patch('gpu_worker_pool.http_client.aiohttp.ClientSession') as mock_session_class:
        # Set up mock HTTP responses
        mock_session = AsyncMock()
        mock_response = AsyncMock()
        mock_response.json.return_value = {
            "gpu_count": 1,
            "total_memory_mb": 8192,
            "total_used_memory_mb": 2048,
            "average_utilization_percent": 25.0,
            "gpus_summary": [
                {"gpu_id": 0, "name": "Factory-GPU", "memory_usage_percent": 25.0, "utilization_percent": 30.0}
            ],
            "total_memory_usage_percent": 25.0,
            "timestamp": datetime.now().isoformat()
        }
        mock_response.status = 200
        mock_session.get.return_value.__aenter__.return_value = mock_response
        mock_session_class.return_value = mock_session
        
        # Use factory function with custom parameters
        async with gpu_worker_pool_client(
            memory_threshold=75.0,
            utilization_threshold=85.0
        ) as client:
            print("Client created via factory function")
            
            # Wait for initial stats
            await asyncio.sleep(0.2)
            
            # Get detailed metrics
            metrics = client.get_detailed_metrics()
            print(f"Detailed metrics: {metrics['total_gpus']} GPUs, thresholds: {metrics['thresholds']}")
            
            assignment = await client.request_gpu()
            print(f"GPU {assignment.gpu_id} assigned via factory client")
            
            await client.release_gpu(assignment)
            print("GPU released")


async def example_gpu_context_manager():
    """Example using the GPU context manager for automatic assignment/release."""
    print("\n=== GPU Context Manager Usage Example ===")
    
    # Mock the HTTP client for this example
    with patch('gpu_worker_pool.http_client.aiohttp.ClientSession') as mock_session_class:
        # Set up mock HTTP responses
        mock_session = AsyncMock()
        mock_response = AsyncMock()
        mock_response.json.return_value = {
            "gpu_count": 2,
            "total_memory_mb": 16384,
            "total_used_memory_mb": 4096,
            "average_utilization_percent": 25.0,
            "gpus_summary": [
                {"gpu_id": 0, "name": "GPU-0", "memory_usage_percent": 20.0, "utilization_percent": 15.0},
                {"gpu_id": 1, "name": "GPU-1", "memory_usage_percent": 30.0, "utilization_percent": 35.0}
            ],
            "total_memory_usage_percent": 25.0,
            "timestamp": datetime.now().isoformat()
        }
        mock_response.status = 200
        mock_session.get.return_value.__aenter__.return_value = mock_response
        mock_session_class.return_value = mock_session
        
        async with GPUWorkerPoolClient() as client:
            print("Using GPU context manager for automatic assignment/release")
            
            # Wait for initial stats
            await asyncio.sleep(0.2)
            
            # Use GPU context manager
            async with GPUContextManager(client) as gpu_id:
                print(f"Automatically assigned GPU {gpu_id}")
                
                # Simulate work on the GPU
                print(f"Performing work on GPU {gpu_id}...")
                await asyncio.sleep(1.0)
                
                # Check status while GPU is in use
                status = client.get_pool_status()
                print(f"Status during work: {status.active_workers} active workers")
            
            print("GPU automatically released by context manager")
            
            # Final status
            status = client.get_pool_status()
            print(f"Final status: {status.active_workers} active workers")


async def example_multiple_workers():
    """Example with multiple concurrent workers."""
    print("\n=== Multiple Workers Example ===")
    
    # Mock the HTTP client for this example
    with patch('gpu_worker_pool.http_client.aiohttp.ClientSession') as mock_session_class:
        # Set up mock HTTP responses
        mock_session = AsyncMock()
        mock_response = AsyncMock()
        mock_response.json.return_value = {
            "gpu_count": 2,
            "total_memory_mb": 16384,
            "total_used_memory_mb": 4096,
            "average_utilization_percent": 25.0,
            "gpus_summary": [
                {"gpu_id": 0, "name": "GPU-0", "memory_usage_percent": 20.0, "utilization_percent": 15.0},
                {"gpu_id": 1, "name": "GPU-1", "memory_usage_percent": 30.0, "utilization_percent": 35.0}
            ],
            "total_memory_usage_percent": 25.0,
            "timestamp": datetime.now().isoformat()
        }
        mock_response.status = 200
        mock_session.get.return_value.__aenter__.return_value = mock_response
        mock_session_class.return_value = mock_session
        
        async def worker_task(client, worker_id):
            """Simulate a worker task."""
            print(f"Worker {worker_id} starting...")
            
            async with GPUContextManager(client) as gpu_id:
                print(f"Worker {worker_id} got GPU {gpu_id}")
                
                # Simulate different work durations
                work_duration = 0.5 + (worker_id * 0.3)
                await asyncio.sleep(work_duration)
                
                print(f"Worker {worker_id} finished work on GPU {gpu_id}")
        
        async with GPUWorkerPoolClient() as client:
            print("Starting multiple concurrent workers...")
            
            # Wait for initial stats
            await asyncio.sleep(0.2)
            
            # Start multiple workers concurrently
            tasks = [
                worker_task(client, i) 
                for i in range(3)  # 3 workers, 2 GPUs - one will be queued
            ]
            
            await asyncio.gather(*tasks)
            print("All workers completed")


async def example_multi_endpoint_basic():
    """Example using multi-endpoint configuration."""
    print("\n=== Multi-Endpoint Configuration Example ===")
    
    # Mock the HTTP client for this example
    with patch('gpu_worker_pool.http_client.aiohttp.ClientSession') as mock_session_class:
        # Set up mock HTTP responses
        mock_session = AsyncMock()
        mock_response = AsyncMock()
        mock_response.json.return_value = {
            "gpu_count": 4,
            "total_memory_mb": 32768,
            "total_used_memory_mb": 8192,
            "average_utilization_percent": 25.0,
            "gpus_summary": [
                {"gpu_id": 0, "name": "GPU-0", "memory_usage_percent": 20.0, "utilization_percent": 15.0},
                {"gpu_id": 1, "name": "GPU-1", "memory_usage_percent": 25.0, "utilization_percent": 30.0},
                {"gpu_id": 2, "name": "GPU-2", "memory_usage_percent": 30.0, "utilization_percent": 25.0},
                {"gpu_id": 3, "name": "GPU-3", "memory_usage_percent": 25.0, "utilization_percent": 30.0}
            ],
            "total_memory_usage_percent": 25.0,
            "timestamp": datetime.now().isoformat()
        }
        mock_response.status = 200
        mock_session.get.return_value.__aenter__.return_value = mock_response
        mock_session_class.return_value = mock_session
        
        # Use multi-endpoint configuration
        async with GPUWorkerPoolClient(
            service_endpoints="http://gpu-server-1:8000,http://gpu-server-2:8000",
            load_balancing_strategy="availability"
        ) as client:
            print("Client started in multi-endpoint mode")
            
            # Wait for initial stats
            await asyncio.sleep(0.2)
            
            # Check if in multi-endpoint mode
            if client.is_multi_endpoint_mode():
                print("Multi-endpoint mode confirmed")
                
                # Get pool status (works same as single-endpoint)
                status = client.get_pool_status()
                print(f"Total GPUs across all endpoints: {status.total_gpus}")
                print(f"Available GPUs: {status.available_gpus}")
                
                # Request GPU - returns global GPU ID in multi-endpoint mode
                assignment = await client.request_gpu()
                print(f"Assigned GPU: {assignment.gpu_id}")
                
                await asyncio.sleep(0.5)  # Simulate work
                
                await client.release_gpu(assignment)
                print("GPU released")
        
        print("Multi-endpoint client stopped automatically")


async def main():
    """Run all examples."""
    print("GPU Worker Pool Client Examples")
    print("=" * 50)
    
    try:
        await example_basic_usage()
        await example_context_manager_usage()
        await example_factory_function()
        await example_gpu_context_manager()
        await example_multiple_workers()
        
        # Only run multi-endpoint example if supported
        if has_multi_endpoint:
            await example_multi_endpoint_basic()
        
        print("\n" + "=" * 50)
        print("All examples completed successfully!")
        
        if has_multi_endpoint:
            print("\nFor more multi-endpoint examples, see: examples/multi_endpoint_usage.py")
        else:
            print("\nFor multi-endpoint examples, run: PYTHONPATH=. python examples/basic_client_usage.py")
        
    except Exception as e:
        print(f"Error running examples: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())