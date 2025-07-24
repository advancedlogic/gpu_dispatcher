#!/usr/bin/env python3
"""
Multi-endpoint usage example for the GPU Worker Pool Client.

This example demonstrates how to use the GPUWorkerPoolClient with multiple
GPU statistics servers for load balancing, failover, and increased capacity.

Note: Run this example with PYTHONPATH=. to use the local code:
    PYTHONPATH=. python examples/multi_endpoint_usage.py
    
Or install the package in development mode:
    pip install -e .
"""

import asyncio
import logging
import inspect
from datetime import datetime
from typing import List
from unittest.mock import AsyncMock, patch

from gpu_worker_pool.client import GPUWorkerPoolClient, gpu_worker_pool_client, GPUContextManager

# Verify that we have the multi-endpoint version
def check_multi_endpoint_support():
    """Check if the current version supports multi-endpoint features."""
    sig = inspect.signature(GPUWorkerPoolClient.__init__)
    if 'service_endpoints' not in sig.parameters:
        print("ERROR: Multi-endpoint support not available!")
        print("Please run with: PYTHONPATH=. python examples/multi_endpoint_usage.py")
        print("Or install in development mode: pip install -e .")
        exit(1)

check_multi_endpoint_support()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


async def example_basic_multi_endpoint():
    """Example of basic multi-endpoint configuration and usage."""
    print("\n=== Basic Multi-Endpoint Usage ===")
    
    # Configure multiple endpoints
    endpoints = "http://localhost:8000" #comma separated endpoints
    
    # Create client with multi-endpoint support
    client = GPUWorkerPoolClient(
        service_endpoints=endpoints,
        load_balancing_strategy="availability",  # Choose endpoint with most available GPUs
        memory_threshold=75.0,
        utilization_threshold=85.0
    )
    
    # Mock the HTTP responses for demonstration
    with patch('gpu_worker_pool.http_client.aiohttp.ClientSession') as mock_session_class:
        # Set up proper mock HTTP responses
        mock_session = AsyncMock()
        mock_response = AsyncMock()
        mock_response.status = 200  # Set as integer, not AsyncMock
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
        # Create async context manager mock
        async_context_manager = AsyncMock()
        async_context_manager.__aenter__.return_value = mock_response
        async_context_manager.__aexit__.return_value = None
        mock_session.get.return_value = async_context_manager
        mock_session_class.return_value = mock_session
        
        try:
            await client.start()
            print(f"Client started in multi-endpoint mode: {client.is_multi_endpoint_mode()}")
            
            # Simulate some work
            await asyncio.sleep(0.5)
            
            # Get endpoint information
            endpoints_info = client.get_endpoints_info()
            if endpoints_info:
                print("\nConnected Endpoints:")
                for endpoint in endpoints_info:
                    print(f"  {endpoint['endpoint_id']}: {endpoint['url']}")
                    print(f"    Status: {'Healthy' if endpoint['is_healthy'] else 'Degraded'}")
                    print(f"    GPUs: {endpoint['available_gpus']}/{endpoint['total_gpus']} available")
                    print(f"    Response time: {endpoint['response_time_ms']}ms")
            
            # Get pool status (aggregated across all endpoints)
            status = client.get_pool_status()
            print(f"\nAggregated Pool Status:")
            print(f"  Total endpoints: {status.total_endpoints}")
            print(f"  Healthy endpoints: {status.healthy_endpoints}")
            print(f"  Total GPUs: {status.total_gpus}")
            print(f"  Available GPUs: {status.available_gpus}")
            
        finally:
            await client.stop()
            print("\nClient stopped.")


async def example_load_balancing_strategies():
    """Example showing different load balancing strategies."""
    print("\n=== Load Balancing Strategies Example ===")
    
    endpoints = "http://localhost:8000" #"http://gpu-1:8000,http://gpu-2:8000,http://gpu-3:8000"
    
    # Example 1: Availability-based (default and recommended)
    print("\n1. Availability-Based Load Balancing:")
    print("   - Selects endpoint with highest percentage of available GPUs")
    print("   - Best for maximizing resource utilization")
    
    async with gpu_worker_pool_client(
        service_endpoints=endpoints,
        load_balancing_strategy="availability"
    ) as client:
        # The load balancer will automatically select the best endpoint
        pass
    
    # Example 2: Round-Robin
    print("\n2. Round-Robin Load Balancing:")
    print("   - Distributes requests evenly across all healthy endpoints")
    print("   - Best for equal distribution")
    
    async with gpu_worker_pool_client(
        service_endpoints=endpoints,
        load_balancing_strategy="round_robin"
    ) as client:
        # Requests will be distributed in circular order
        pass
    
    # Example 3: Weighted
    print("\n3. Weighted Load Balancing:")
    print("   - Distributes based on total GPU capacity of each endpoint")
    print("   - Best for heterogeneous clusters")
    
    async with gpu_worker_pool_client(
        service_endpoints=endpoints,
        load_balancing_strategy="weighted"
    ) as client:
        # Endpoints with more GPUs get more requests
        pass


async def example_gpu_assignment_with_global_ids():
    """Example showing GPU assignment with global IDs."""
    print("\n=== Global GPU ID Assignment Example ===")
    
    client = GPUWorkerPoolClient(
        service_endpoints= "http://localhost:8000", #"http://server1:8000,http://server2:8000",
        load_balancing_strategy="availability"
    )
    
    # Mock setup for demonstration
    with patch('gpu_worker_pool.http_client.aiohttp.ClientSession') as mock_session_class:
        # Set up proper mock HTTP responses
        mock_session = AsyncMock()
        mock_response = AsyncMock()
        mock_response.status = 200
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
        # Create async context manager mock
        async_context_manager = AsyncMock()
        async_context_manager.__aenter__.return_value = mock_response
        async_context_manager.__aexit__.return_value = None
        mock_session.get.return_value = async_context_manager
        mock_session_class.return_value = mock_session
        
        async with client:
            await asyncio.sleep(0.5)  # Let client initialize
            
            # In multi-endpoint mode, GPU assignments use global IDs
            print("\nRequesting GPU from multi-endpoint pool...")
            
            # Mock a GPU assignment for demonstration
            # In a real scenario, this would come from client.request_gpu()
            mock_assignment = {
                'global_gpu_id': "server1:2",
                'endpoint_id': "server1",
                'local_gpu_id': 2,
                'worker_id': "worker-demo",
                'assigned_at': datetime.now()
            }
            
            print(f"Assigned GPU: {mock_assignment['global_gpu_id']}")
            print(f"  Endpoint: {mock_assignment['endpoint_id']}")
            print(f"  Local GPU ID: {mock_assignment['local_gpu_id']}")
            print(f"  Worker: {mock_assignment['worker_id']}")
            
            # Simulate work
            print("\nPerforming work on GPU...")
            await asyncio.sleep(1.0)
            
            # Release would automatically route to correct endpoint
            print(f"Releasing GPU {mock_assignment['global_gpu_id']}")


async def example_failover_and_recovery():
    """Example demonstrating failover and recovery capabilities."""
    print("\n=== Failover and Recovery Example ===")
    
    client = GPUWorkerPoolClient(
        service_endpoints= "http://localhost:8000", #"http://server1:8000,http://server2:8000,http://server3:8000",
        load_balancing_strategy="availability"
    )
    
    with patch('gpu_worker_pool.http_client.aiohttp.ClientSession') as mock_session_class:
        # Set up proper mock HTTP responses
        mock_session = AsyncMock()
        mock_response = AsyncMock()
        mock_response.status = 200
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
        # Create async context manager mock
        async_context_manager = AsyncMock()
        async_context_manager.__aenter__.return_value = mock_response
        async_context_manager.__aexit__.return_value = None
        mock_session.get.return_value = async_context_manager
        mock_session_class.return_value = mock_session
        
        async with client:
            await asyncio.sleep(0.5)
            
            # Check initial health status
            print("\nInitial Error Recovery Status:")
            recovery_status = client.get_error_recovery_status()
            print(f"  Healthy endpoints: {len(recovery_status['degradation_manager']['healthy_endpoints'])}")
            print(f"  Degraded endpoints: {len(recovery_status['degradation_manager']['degraded_endpoints'])}")
            
            # Simulate endpoint failure (in real scenario, this happens automatically)
            print("\n[Simulating server2 failure...]")
            
            # Check circuit breaker status
            print("\nCircuit Breaker Status:")
            for endpoint_id, cb_stats in recovery_status['degradation_manager']['circuit_breaker_stats'].items():
                print(f"  {endpoint_id}: {cb_stats['state']} (failure rate: {cb_stats['failure_rate']}%)")
            
            # Manual recovery trigger (optional - system does this automatically)
            print("\n[Triggering manual recovery for server2...]")
            try:
                success = await client.trigger_endpoint_recovery("server2")
                if success:
                    print("Recovery triggered successfully")
            except Exception as e:
                print(f"Recovery trigger failed: {e}")
            
            # Print formatted recovery summary
            print("\nFormatted Recovery Summary:")
            client.print_error_recovery_summary()


async def example_monitoring_and_metrics():
    """Example showing comprehensive monitoring capabilities."""
    print("\n=== Monitoring and Metrics Example ===")
    
    client = GPUWorkerPoolClient(
        service_endpoints= "http://localhost:8000", #"http://gpu1:8000,http://gpu2:8000,http://gpu3:8000",
        load_balancing_strategy="availability"
    )
    
    with patch('gpu_worker_pool.http_client.aiohttp.ClientSession') as mock_session_class:
        # Set up proper mock HTTP responses
        mock_session = AsyncMock()
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = {
            "gpu_count": 3,
            "total_memory_mb": 24576,
            "total_used_memory_mb": 6144,
            "average_utilization_percent": 30.0,
            "gpus_summary": [
                {"gpu_id": 0, "name": "GPU-0", "memory_usage_percent": 25.0, "utilization_percent": 30.0},
                {"gpu_id": 1, "name": "GPU-1", "memory_usage_percent": 20.0, "utilization_percent": 25.0},
                {"gpu_id": 2, "name": "GPU-2", "memory_usage_percent": 30.0, "utilization_percent": 35.0}
            ],
            "total_memory_usage_percent": 25.0,
            "timestamp": datetime.now().isoformat()
        }
        # Create async context manager mock
        async_context_manager = AsyncMock()
        async_context_manager.__aenter__.return_value = mock_response
        async_context_manager.__aexit__.return_value = None
        mock_session.get.return_value = async_context_manager
        mock_session_class.return_value = mock_session
        
        async with client:
            await asyncio.sleep(0.5)
            
            # Get detailed metrics
            metrics = client.get_detailed_metrics()
            
            print("\nDetailed Metrics:")
            print(f"Total GPUs: {metrics['total_gpus']}")
            print(f"Available GPUs: {metrics['available_gpus']}")
            print(f"Active Workers: {metrics['active_workers']}")
            
            # Endpoint-specific metrics
            if 'endpoints' in metrics:
                print("\nPer-Endpoint Metrics:")
                endpoint_details = metrics['endpoints'].get('details', {})
                for endpoint_id, endpoint_metrics in endpoint_details.items():
                    print(f"\n  {endpoint_id}:")
                    print(f"    Total GPUs: {endpoint_metrics.get('total_gpus', 0)}")
                    print(f"    Available GPUs: {endpoint_metrics.get('available_gpus', 0)}")
                    print(f"    Health: {'Healthy' if endpoint_metrics.get('is_healthy', False) else 'Degraded'}")
                    print(f"    Response Time: {endpoint_metrics.get('avg_response_time_ms', 0)}ms")
            
            # Load balancer statistics
            if 'load_balancer' in metrics:
                lb_stats = metrics['load_balancer']
                print(f"\nLoad Balancer Statistics:")
                print(f"  Strategy: {lb_stats.get('strategy_name', 'unknown')}")
                print(f"  Total Requests: {lb_stats.get('total_requests', 0)}")
                print(f"  Success Rate: {lb_stats.get('success_rate', 0)}%")


async def example_graceful_degradation():
    """Example showing graceful degradation when endpoints fail."""
    print("\n=== Graceful Degradation Example ===")
    
    client = GPUWorkerPoolClient(
        service_endpoints= "http://localhost:8000", #"http://server1:8000,http://server2:8000",
        load_balancing_strategy="availability"
    )
    
    with patch('gpu_worker_pool.http_client.aiohttp.ClientSession') as mock_session_class:
        # Set up proper mock HTTP responses
        mock_session = AsyncMock()
        mock_response = AsyncMock()
        mock_response.status = 200  # Set as integer, not AsyncMock
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
        # Create async context manager mock
        async_context_manager = AsyncMock()
        async_context_manager.__aenter__.return_value = mock_response
        async_context_manager.__aexit__.return_value = None
        mock_session.get.return_value = async_context_manager
        mock_session_class.return_value = mock_session
        
        async with client:
            await asyncio.sleep(0.5)
            
            print("\nScenario: All endpoints become unavailable")
            print("The system will queue requests until endpoints recover")
            
            # Queue a request for retry when endpoints become available
            async def gpu_request_with_retry():
                print("Attempting to request GPU...")
                # In real scenario, this would wait for endpoint recovery
                await asyncio.sleep(0.5)
                return "gpu-assignment-when-available"
            
            try:
                print("\nQueueing request for retry...")
                # This would normally block until an endpoint becomes available
                result = await client.queue_request_for_retry(gpu_request_with_retry)
                print(f"Request completed: {result}")
            except Exception as e:
                print(f"Request failed: {e}")


async def example_migration_from_single_endpoint():
    """Example showing migration from single to multi-endpoint."""
    print("\n=== Migration Example ===")
    
    print("Before (single-endpoint):")
    print('client = GPUWorkerPoolClient(service_endpoint="http://gpu-server:8000")')
    
    print("\nAfter (multi-endpoint):")
    print('client = GPUWorkerPoolClient(service_endpoints="http://gpu-1:8000,http://gpu-2:8000")')
    
    print("\nKey differences:")
    print("1. Use 'service_endpoints' (plural) instead of 'service_endpoint'")
    print("2. Provide comma-separated list of endpoints")
    print("3. GPU IDs become global (e.g., 'server1:0' instead of just '0')")
    print("4. All other code remains the same!")
    
    # Demonstrate backward compatibility
    print("\nBackward Compatibility:")
    print("- Single endpoint in 'service_endpoints' works like single-endpoint mode")
    print("- Environment variables are backward compatible")
    print("- API methods work the same way")


async def main():
    """Run all multi-endpoint examples."""
    print("GPU Worker Pool Multi-Endpoint Examples")
    print("=" * 50)
    
    try:
        await example_basic_multi_endpoint()
        await example_load_balancing_strategies()
        await example_gpu_assignment_with_global_ids()
        await example_failover_and_recovery()
        await example_monitoring_and_metrics()
        await example_graceful_degradation()
        await example_migration_from_single_endpoint()
        
        print("\n" + "=" * 50)
        print("All multi-endpoint examples completed successfully!")
        
    except Exception as e:
        print(f"Error running examples: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())