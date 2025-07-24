#!/usr/bin/env python3
"""
Simple demonstration of multi-endpoint configuration and usage.

This example shows the configuration patterns without requiring actual GPU servers.
"""

import asyncio
from gpu_worker_pool.client import GPUWorkerPoolClient

# Example configurations for different scenarios

def show_configuration_examples():
    """Show various multi-endpoint configuration examples."""
    
    print("GPU Worker Pool Multi-Endpoint Configuration Examples")
    print("=" * 60)
    
    print("\n1. Basic Multi-Endpoint Configuration:")
    print("   Using multiple GPU servers for increased capacity")
    print("""
    client = GPUWorkerPoolClient(
        service_endpoints="http://gpu-1:8000,http://gpu-2:8000,http://gpu-3:8000",
        load_balancing_strategy="availability",  # Choose best available endpoint
        memory_threshold=75.0,
        utilization_threshold=85.0
    )
    """)
    
    print("\n2. Environment Variable Configuration:")
    print("   Configure via environment for production deployments")
    print("""
    export GPU_STATS_SERVICE_ENDPOINTS="http://gpu-1:8000,http://gpu-2:8000,http://gpu-3:8000"
    export GPU_LOAD_BALANCING_STRATEGY="availability"
    export GPU_MEMORY_THRESHOLD_PERCENT="75.0"
    export GPU_UTILIZATION_THRESHOLD_PERCENT="85.0"
    """)
    
    print("\n3. Load Balancing Strategies:")
    print("\n   a) Availability-Based (default, recommended):")
    print("      - Selects endpoint with highest percentage of available GPUs")
    print("      - Best for maximizing resource utilization")
    print("""
    client = GPUWorkerPoolClient(
        service_endpoints="...",
        load_balancing_strategy="availability"
    )
    """)
    
    print("\n   b) Round-Robin:")
    print("      - Distributes requests evenly across healthy endpoints")
    print("      - Best for equal distribution and testing")
    print("""
    client = GPUWorkerPoolClient(
        service_endpoints="...",
        load_balancing_strategy="round_robin"
    )
    """)
    
    print("\n   c) Weighted:")
    print("      - Distributes based on total GPU capacity")
    print("      - Best for heterogeneous clusters")
    print("""
    client = GPUWorkerPoolClient(
        service_endpoints="...",
        load_balancing_strategy="weighted"
    )
    """)
    
    print("\n4. High Availability Configuration:")
    print("   Primary + backup servers for fault tolerance")
    print("""
    client = GPUWorkerPoolClient(
        service_endpoints="http://primary-gpu:8000,http://backup-gpu:8000",
        load_balancing_strategy="availability"
    )
    """)
    
    print("\n5. Geographic Distribution:")
    print("   Multi-region GPU servers")
    print("""
    client = GPUWorkerPoolClient(
        service_endpoints="http://us-east:8000,http://us-west:8000,http://eu:8000",
        load_balancing_strategy="availability"
    )
    """)


def show_usage_patterns():
    """Show common usage patterns."""
    
    print("\n\nCommon Usage Patterns")
    print("=" * 60)
    
    print("\n1. Basic Usage (same as single-endpoint):")
    print("""
    async with GPUWorkerPoolClient(service_endpoints="...") as client:
        # Request GPU - returns global GPU ID in multi-endpoint mode
        assignment = await client.request_gpu()
        print(f"Assigned GPU: {assignment.gpu_id}")  # e.g., "server1:2"
        
        # Do work with GPU...
        
        # Release GPU - automatically routes to correct endpoint
        await client.release_gpu(assignment)
    """)
    
    print("\n2. Checking Multi-Endpoint Status:")
    print("""
    async with GPUWorkerPoolClient(service_endpoints="...") as client:
        # Check if in multi-endpoint mode
        if client.is_multi_endpoint_mode():
            # Get endpoint information
            endpoints = client.get_endpoints_info()
            for endpoint in endpoints:
                print(f"{endpoint['endpoint_id']}: {endpoint['is_healthy']}")
            
            # Get aggregated pool status
            status = client.get_pool_status()
            print(f"Total endpoints: {status.total_endpoints}")
            print(f"Healthy endpoints: {status.healthy_endpoints}")
            print(f"Total GPUs: {status.total_gpus}")
    """)
    
    print("\n3. Error Recovery Monitoring:")
    print("""
    async with GPUWorkerPoolClient(service_endpoints="...") as client:
        # Get error recovery status
        recovery_status = client.get_error_recovery_status()
        print(f"Healthy: {len(recovery_status['degradation_manager']['healthy_endpoints'])}")
        print(f"Degraded: {len(recovery_status['degradation_manager']['degraded_endpoints'])}")
        
        # Print formatted summary
        client.print_error_recovery_summary()
        
        # Manual recovery trigger (optional)
        await client.trigger_endpoint_recovery("server2")
    """)
    
    print("\n4. Monitoring and Metrics:")
    print("""
    async with GPUWorkerPoolClient(service_endpoints="...") as client:
        # Get detailed metrics
        metrics = client.get_detailed_metrics()
        
        # Overall statistics
        print(f"Total GPUs: {metrics['total_gpus']}")
        print(f"Available GPUs: {metrics['available_gpus']}")
        
        # Per-endpoint metrics
        for endpoint_id, stats in metrics['endpoints']['details'].items():
            print(f"{endpoint_id}: {stats['available_gpus']}/{stats['total_gpus']} GPUs")
        
        # Load balancer statistics
        lb_stats = metrics['load_balancer']
        print(f"Strategy: {lb_stats['strategy_name']}")
        print(f"Success rate: {lb_stats['success_rate']}%")
    """)


def show_migration_guide():
    """Show migration from single to multi-endpoint."""
    
    print("\n\nMigration from Single to Multi-Endpoint")
    print("=" * 60)
    
    print("\nBefore (single-endpoint):")
    print("""
    client = GPUWorkerPoolClient(
        service_endpoint="http://gpu-server:8000"
    )
    """)
    
    print("\nAfter (multi-endpoint):")
    print("""
    client = GPUWorkerPoolClient(
        service_endpoints="http://gpu-server-1:8000,http://gpu-server-2:8000"
    )
    """)
    
    print("\nKey differences:")
    print("- Use 'service_endpoints' (plural) instead of 'service_endpoint'")
    print("- GPU IDs become global (e.g., 'server1:0' instead of '0')")
    print("- Additional monitoring methods available")
    print("- All existing code continues to work!")


def show_troubleshooting():
    """Show common troubleshooting tips."""
    
    print("\n\nTroubleshooting Tips")
    print("=" * 60)
    
    print("\n1. Check if in multi-endpoint mode:")
    print("""
    if client.is_multi_endpoint_mode():
        print("Multi-endpoint mode active")
    else:
        print("Single-endpoint mode")
    """)
    
    print("\n2. Verify endpoint health:")
    print("""
    endpoints = client.get_endpoints_info()
    for endpoint in endpoints:
        if not endpoint['is_healthy']:
            print(f"Unhealthy endpoint: {endpoint['endpoint_id']}")
    """)
    
    print("\n3. Check circuit breaker status:")
    print("""
    status = client.get_error_recovery_status()
    for endpoint_id, cb_stats in status['degradation_manager']['circuit_breaker_stats'].items():
        if cb_stats['state'] == 'open':
            print(f"Circuit breaker open for {endpoint_id}")
    """)
    
    print("\n4. Common issues:")
    print("- Wrong parameter name: use 'service_endpoints' not 'service_endpoint'")
    print("- Wrong strategy name: use 'availability' not 'availability_based'")
    print("- Environment variable: use 'GPU_STATS_SERVICE_ENDPOINTS' not 'GPU_SERVICE_ENDPOINTS'")


def main():
    """Run all demonstrations."""
    show_configuration_examples()
    show_usage_patterns()
    show_migration_guide()
    show_troubleshooting()
    
    print("\n\nFor working examples, see:")
    print("- examples/basic_client_usage.py")
    print("- examples/multi_endpoint_usage.py")
    print("\nFor detailed documentation, see:")
    print("- docs/API.md")
    print("- docs/MIGRATION_GUIDE.md")
    print("- docs/TROUBLESHOOTING.md")


if __name__ == "__main__":
    main()