#!/usr/bin/env python3
"""Test script to verify the GPU Worker Pool package works correctly."""

import asyncio
import sys

def test_imports():
    """Test that all public API imports work correctly."""
    print("Testing imports...")
    
    try:
        # Test main imports
        from gpu_worker_pool import (
            GPUWorkerPoolClient,
            GPUContextManager,
            gpu_worker_pool_client,
            GPUAssignment,
            GPUInfo,
            GPUStats,
            PoolStatus,
            WorkerInfo,
            EnvironmentConfigurationManager,
            WorkerTimeoutError,
            StaleAssignmentError,
            ServiceUnavailableError,
            __version__
        )
        
        print(f"✅ All imports successful")
        print(f"✅ Package version: {__version__}")
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False


def test_basic_functionality():
    """Test basic functionality without requiring a real GPU service."""
    print("\nTesting basic functionality...")
    
    try:
        from gpu_worker_pool import GPUWorkerPoolClient, EnvironmentConfigurationManager
        
        # Test configuration
        config = EnvironmentConfigurationManager()
        print(f"✅ Configuration manager created")
        print(f"   - Service endpoint: {config.get_service_endpoint()}")
        print(f"   - Memory threshold: {config.get_memory_threshold()}%")
        print(f"   - Utilization threshold: {config.get_utilization_threshold()}%")
        
        # Test client creation (without starting)
        client = GPUWorkerPoolClient()
        print(f"✅ Client created successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        return False


async def test_async_functionality():
    """Test async functionality."""
    print("\nTesting async functionality...")
    
    try:
        from gpu_worker_pool import GPUWorkerPoolClient
        
        # Test client creation and basic properties
        client = GPUWorkerPoolClient(
            service_endpoint="http://test:8080",
            memory_threshold=75.0,
            utilization_threshold=85.0
        )
        
        print(f"✅ Async client created with custom config")
        print(f"   - Memory threshold: {client.memory_threshold}")
        print(f"   - Utilization threshold: {client.utilization_threshold}")
        print(f"   - Worker timeout: {client.worker_timeout}")
        
        return True
        
    except Exception as e:
        print(f"❌ Async functionality test failed: {e}")
        return False


def test_data_models():
    """Test data model creation and validation."""
    print("\nTesting data models...")
    
    try:
        from gpu_worker_pool import GPUInfo, GPUStats, GPUAssignment, PoolStatus, WorkerInfo
        from datetime import datetime
        
        # Test GPUInfo
        gpu_info = GPUInfo(
            gpu_id=0,
            name="Test GPU",
            memory_usage_percent=50.0,
            utilization_percent=25.0
        )
        print(f"✅ GPUInfo created: {gpu_info.name}")
        
        # Test GPUStats
        gpu_stats = GPUStats(
            gpu_count=1,
            total_memory_mb=8192,
            total_used_memory_mb=4096,
            average_utilization_percent=25.0,
            gpus_summary=[gpu_info],
            total_memory_usage_percent=50.0,
            timestamp=datetime.now().isoformat()
        )
        print(f"✅ GPUStats created: {gpu_stats.gpu_count} GPUs")
        
        # Test GPUAssignment
        assignment = GPUAssignment(
            gpu_id=0,
            worker_id="test-worker",
            assigned_at=datetime.now()
        )
        print(f"✅ GPUAssignment created: GPU {assignment.gpu_id}")
        
        # Test PoolStatus
        status = PoolStatus(
            total_gpus=1,
            available_gpus=1,
            active_workers=0,
            blocked_workers=0
        )
        print(f"✅ PoolStatus created: {status.total_gpus} total GPUs")
        
        return True
        
    except Exception as e:
        print(f"❌ Data models test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("GPU Worker Pool Package Test")
    print("=" * 40)
    
    tests = [
        test_imports,
        test_basic_functionality,
        test_data_models,
    ]
    
    async_tests = [
        test_async_functionality,
    ]
    
    # Run synchronous tests
    results = []
    for test in tests:
        results.append(test())
    
    # Run asynchronous tests
    for test in async_tests:
        results.append(asyncio.run(test()))
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print(f"\n" + "=" * 40)
    print(f"Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All tests passed! Package is working correctly.")
        return 0
    else:
        print("❌ Some tests failed. Please check the output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())