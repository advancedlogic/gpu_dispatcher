#!/usr/bin/env python3
"""Simple test script to verify MultiEndpointGPUMonitor implementation."""

import asyncio
import sys
from datetime import datetime
from typing import List

# Add the gpu_worker_pool to the path
sys.path.insert(0, '.')

from gpu_worker_pool.multi_endpoint_gpu_monitor import MultiEndpointGPUMonitor, MockMultiEndpointGPUMonitor
from gpu_worker_pool.models import GlobalGPUInfo, EndpointInfo, GPUStats


def test_mock_multi_endpoint_gpu_monitor():
    """Test the mock multi-endpoint GPU monitor."""
    print("Testing MockMultiEndpointGPUMonitor...")
    
    # Create mock global GPU data
    mock_global_gpus = [
        GlobalGPUInfo(
            global_gpu_id="server1:0",
            endpoint_id="server1",
            local_gpu_id=0,
            name="RTX 4090",
            memory_usage_percent=45.0,
            utilization_percent=30.0,
            is_available=True
        ),
        GlobalGPUInfo(
            global_gpu_id="server2:0",
            endpoint_id="server2",
            local_gpu_id=0,
            name="RTX 4080",
            memory_usage_percent=60.0,
            utilization_percent=75.0,
            is_available=False
        )
    ]
    
    # Create mock endpoints
    mock_endpoints = [
        EndpointInfo(
            endpoint_id="server1",
            url="http://server1:8000",
            is_healthy=True,
            last_seen=datetime.now(),
            total_gpus=1,
            available_gpus=1,
            response_time_ms=150.0
        ),
        EndpointInfo(
            endpoint_id="server2",
            url="http://server2:8000",
            is_healthy=True,
            last_seen=datetime.now(),
            total_gpus=1,
            available_gpus=0,
            response_time_ms=200.0
        )
    ]
    
    # Create mock monitor
    monitor = MockMultiEndpointGPUMonitor(
        mock_global_gpus=mock_global_gpus,
        mock_endpoints=mock_endpoints
    )
    
    # Test basic functionality
    current_stats = monitor.get_current_stats()
    assert current_stats is not None, "Should have current stats"
    assert current_stats.gpu_count == 2, f"Expected 2 GPUs, got {current_stats.gpu_count}"
    
    global_gpus = monitor.get_current_global_gpus()
    assert global_gpus is not None, "Should have global GPU data"
    assert len(global_gpus) == 2, f"Expected 2 global GPUs, got {len(global_gpus)}"
    
    # Test endpoint health status
    health_status = monitor.get_endpoint_health_status()
    assert isinstance(health_status, dict), "Health status should be a dictionary"
    
    # Test monitor status
    monitor_status = monitor.get_monitor_status()
    assert isinstance(monitor_status, dict), "Monitor status should be a dictionary"
    assert "is_running" in monitor_status, "Monitor status should include is_running"
    assert "total_global_gpus" in monitor_status, "Monitor status should include total_global_gpus"
    
    print("✓ MockMultiEndpointGPUMonitor tests passed")


async def test_mock_monitor_lifecycle():
    """Test the mock monitor start/stop lifecycle."""
    print("Testing mock monitor lifecycle...")
    
    monitor = MockMultiEndpointGPUMonitor()
    
    # Test initial state
    assert not monitor._is_running, "Monitor should not be running initially"
    assert monitor.start_call_count == 0, "Start should not have been called"
    
    # Test start
    await monitor.start()
    assert monitor._is_running, "Monitor should be running after start"
    assert monitor.start_call_count == 1, "Start should have been called once"
    
    # Test stop
    await monitor.stop()
    assert not monitor._is_running, "Monitor should not be running after stop"
    assert monitor.stop_call_count == 1, "Stop should have been called once"
    
    print("✓ Mock monitor lifecycle tests passed")


async def test_callback_system():
    """Test the callback system."""
    print("Testing callback system...")
    
    callback_called = False
    received_stats = None
    
    def test_callback(stats: GPUStats):
        nonlocal callback_called, received_stats
        callback_called = True
        received_stats = stats
    
    # Create mock data
    mock_global_gpus = [
        GlobalGPUInfo(
            global_gpu_id="test:0",
            endpoint_id="test",
            local_gpu_id=0,
            name="Test GPU",
            memory_usage_percent=50.0,
            utilization_percent=25.0,
            is_available=True
        )
    ]
    
    monitor = MockMultiEndpointGPUMonitor(mock_global_gpus=mock_global_gpus)
    
    # Register callback
    monitor.on_stats_update(test_callback)
    
    # Start monitor (should trigger callback)
    await monitor.start()
    
    # Verify callback was called
    assert callback_called, "Callback should have been called"
    assert received_stats is not None, "Should have received stats"
    assert received_stats.gpu_count == 1, "Should have received stats for 1 GPU"
    
    # Test removing callback
    monitor.remove_stats_callback(test_callback)
    
    # Reset callback state
    callback_called = False
    received_stats = None
    
    # Trigger update (callback should not be called)
    await monitor.trigger_stats_update(mock_global_gpus)
    
    assert not callback_called, "Callback should not have been called after removal"
    
    await monitor.stop()
    
    print("✓ Callback system tests passed")


def test_aggregated_stats_creation():
    """Test the creation of aggregated stats from global GPU data."""
    print("Testing aggregated stats creation...")
    
    # Create test data
    global_gpus = [
        GlobalGPUInfo(
            global_gpu_id="server1:0",
            endpoint_id="server1",
            local_gpu_id=0,
            name="RTX 4090",
            memory_usage_percent=40.0,
            utilization_percent=20.0,
            is_available=True
        ),
        GlobalGPUInfo(
            global_gpu_id="server1:1",
            endpoint_id="server1",
            local_gpu_id=1,
            name="RTX 4090",
            memory_usage_percent=60.0,
            utilization_percent=80.0,
            is_available=False
        ),
        GlobalGPUInfo(
            global_gpu_id="server2:0",
            endpoint_id="server2",
            local_gpu_id=0,
            name="RTX 4080",
            memory_usage_percent=30.0,
            utilization_percent=50.0,
            is_available=True
        )
    ]
    
    monitor = MockMultiEndpointGPUMonitor()
    aggregated_stats = monitor._create_aggregated_stats(global_gpus)
    
    # Verify aggregated statistics
    assert aggregated_stats.gpu_count == 3, f"Expected 3 GPUs, got {aggregated_stats.gpu_count}"
    assert len(aggregated_stats.gpus_summary) == 3, f"Expected 3 GPU summaries, got {len(aggregated_stats.gpus_summary)}"
    
    # Check average utilization (20 + 80 + 50) / 3 = 50
    expected_avg_util = (20.0 + 80.0 + 50.0) / 3
    assert abs(aggregated_stats.average_utilization_percent - expected_avg_util) < 0.1, \
        f"Expected avg utilization {expected_avg_util}, got {aggregated_stats.average_utilization_percent}"
    
    # Check that GPU names include endpoint information
    for gpu_info in aggregated_stats.gpus_summary:
        assert ":" in gpu_info.name, f"GPU name should include endpoint: {gpu_info.name}"
    
    print("✓ Aggregated stats creation tests passed")


async def main():
    """Run all tests."""
    print("Running MultiEndpointGPUMonitor tests...\n")
    
    try:
        # Run synchronous tests
        test_mock_multi_endpoint_gpu_monitor()
        test_aggregated_stats_creation()
        
        # Run asynchronous tests
        await test_mock_monitor_lifecycle()
        await test_callback_system()
        
        print("\n✅ All tests passed!")
        return 0
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)