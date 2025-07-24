#!/usr/bin/env python3
"""Simple test script to verify the core data models work correctly."""

from datetime import datetime
from gpu_worker_pool.models import GPUInfo, GPUStats, WorkerInfo, GPUAssignment, PoolStatus, create_gpu_stats_from_json


def test_gpu_info():
    """Test GPUInfo creation and validation."""
    print("Testing GPUInfo...")
    
    # Valid GPU info
    gpu = GPUInfo(gpu_id=0, name="NVIDIA RTX 4090", memory_usage_percent=45.5, utilization_percent=78.2)
    assert gpu.gpu_id == 0
    assert gpu.name == "NVIDIA RTX 4090"
    assert gpu.memory_usage_percent == 45.5
    assert gpu.utilization_percent == 78.2
    
    # Test validation
    try:
        GPUInfo(gpu_id=-1, name="Test", memory_usage_percent=50, utilization_percent=50)
        assert False, "Should have raised ValueError for negative gpu_id"
    except ValueError:
        pass
    
    try:
        GPUInfo(gpu_id=0, name="", memory_usage_percent=50, utilization_percent=50)
        assert False, "Should have raised ValueError for empty name"
    except ValueError:
        pass
    
    try:
        GPUInfo(gpu_id=0, name="Test", memory_usage_percent=150, utilization_percent=50)
        assert False, "Should have raised ValueError for memory_usage_percent > 100"
    except ValueError:
        pass
    
    print("✓ GPUInfo tests passed")


def test_gpu_stats():
    """Test GPUStats creation and validation."""
    print("Testing GPUStats...")
    
    gpu1 = GPUInfo(gpu_id=0, name="GPU 0", memory_usage_percent=45.5, utilization_percent=78.2)
    gpu2 = GPUInfo(gpu_id=1, name="GPU 1", memory_usage_percent=32.1, utilization_percent=65.4)
    
    stats = GPUStats(
        gpu_count=2,
        total_memory_mb=16384,
        total_used_memory_mb=8192,
        average_utilization_percent=71.8,
        gpus_summary=[gpu1, gpu2],
        total_memory_usage_percent=50.0,
        timestamp="2024-01-01T12:00:00Z"
    )
    
    assert stats.gpu_count == 2
    assert len(stats.gpus_summary) == 2
    assert stats.total_memory_mb == 16384
    
    # Test validation - gpu_count mismatch
    try:
        GPUStats(
            gpu_count=3,  # Mismatch with gpus_summary length
            total_memory_mb=16384,
            total_used_memory_mb=8192,
            average_utilization_percent=71.8,
            gpus_summary=[gpu1, gpu2],
            total_memory_usage_percent=50.0,
            timestamp="2024-01-01T12:00:00Z"
        )
        assert False, "Should have raised ValueError for gpu_count mismatch"
    except ValueError:
        pass
    
    print("✓ GPUStats tests passed")


def test_worker_info():
    """Test WorkerInfo creation and validation."""
    print("Testing WorkerInfo...")
    
    def dummy_callback(gpu_id: int):
        pass
    
    def dummy_error_handler(error: Exception):
        pass
    
    worker = WorkerInfo(
        id="worker-123",
        enqueued_at=datetime.now(),
        callback=dummy_callback,
        on_error=dummy_error_handler
    )
    
    assert worker.id == "worker-123"
    assert isinstance(worker.enqueued_at, datetime)
    assert callable(worker.callback)
    assert callable(worker.on_error)
    
    # Test validation
    try:
        WorkerInfo(
            id="",  # Empty ID
            enqueued_at=datetime.now(),
            callback=dummy_callback,
            on_error=dummy_error_handler
        )
        assert False, "Should have raised ValueError for empty id"
    except ValueError:
        pass
    
    print("✓ WorkerInfo tests passed")


def test_gpu_assignment():
    """Test GPUAssignment creation and validation."""
    print("Testing GPUAssignment...")
    
    assignment = GPUAssignment(
        gpu_id=0,
        worker_id="worker-123",
        assigned_at=datetime.now()
    )
    
    assert assignment.gpu_id == 0
    assert assignment.worker_id == "worker-123"
    assert isinstance(assignment.assigned_at, datetime)
    
    # Test validation
    try:
        GPUAssignment(gpu_id=-1, worker_id="worker-123", assigned_at=datetime.now())
        assert False, "Should have raised ValueError for negative gpu_id"
    except ValueError:
        pass
    
    print("✓ GPUAssignment tests passed")


def test_pool_status():
    """Test PoolStatus creation and validation."""
    print("Testing PoolStatus...")
    
    def dummy_callback(gpu_id: int):
        pass
    
    def dummy_error_handler(error: Exception):
        pass
    
    worker = WorkerInfo(
        id="worker-123",
        enqueued_at=datetime.now(),
        callback=dummy_callback,
        on_error=dummy_error_handler
    )
    
    status = PoolStatus(
        total_gpus=2,
        available_gpus=1,
        active_workers=1,
        blocked_workers=0,
        gpu_assignments={0: [worker]}
    )
    
    assert status.total_gpus == 2
    assert status.available_gpus == 1
    assert status.active_workers == 1
    assert status.blocked_workers == 0
    assert len(status.gpu_assignments[0]) == 1
    
    # Test validation
    try:
        PoolStatus(
            total_gpus=2,
            available_gpus=3,  # More available than total
            active_workers=1,
            blocked_workers=0
        )
        assert False, "Should have raised ValueError for available_gpus > total_gpus"
    except ValueError:
        pass
    
    print("✓ PoolStatus tests passed")


def test_create_gpu_stats_from_json():
    """Test JSON parsing functionality."""
    print("Testing create_gpu_stats_from_json...")
    
    json_data = {
        "gpu_count": 2,
        "total_memory_mb": 16384,
        "total_used_memory_mb": 8192,
        "average_utilization_percent": 71.8,
        "total_memory_usage_percent": 50.0,
        "timestamp": "2024-01-01T12:00:00Z",
        "gpus_summary": [
            {
                "gpu_id": 0,
                "name": "NVIDIA RTX 4090",
                "memory_usage_percent": 45.5,
                "utilization_percent": 78.2
            },
            {
                "gpu_id": 1,
                "name": "NVIDIA RTX 4090",
                "memory_usage_percent": 32.1,
                "utilization_percent": 65.4
            }
        ]
    }
    
    stats = create_gpu_stats_from_json(json_data)
    assert stats.gpu_count == 2
    assert len(stats.gpus_summary) == 2
    assert stats.gpus_summary[0].gpu_id == 0
    assert stats.gpus_summary[0].name == "NVIDIA RTX 4090"
    
    # Test missing field
    try:
        invalid_data = json_data.copy()
        del invalid_data['gpu_count']
        create_gpu_stats_from_json(invalid_data)
        assert False, "Should have raised ValueError for missing field"
    except ValueError:
        pass
    
    print("✓ create_gpu_stats_from_json tests passed")


if __name__ == "__main__":
    print("Running core data model tests...\n")
    
    test_gpu_info()
    test_gpu_stats()
    test_worker_info()
    test_gpu_assignment()
    test_pool_status()
    test_create_gpu_stats_from_json()
    
    print("\n✅ All tests passed! Core data models are working correctly.")