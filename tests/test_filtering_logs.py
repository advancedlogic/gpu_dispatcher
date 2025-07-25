#!/usr/bin/env python3
"""
Test script to verify the filtering logs when GPUs don't exist on the system.
This tests the enhanced logging for missing GPU warnings.
"""

import os
import sys
import logging
import io

# Add the gpu_worker_pool directory to the path
sys.path.insert(0, 'gpu_worker_pool')

def test_filtering_logs():
    """Test the filtering logs for missing GPUs"""
    
    print("Testing GPU filtering logs for missing GPUs...")
    print("=" * 60)
    
    # Set up log capture
    log_capture = io.StringIO()
    handler = logging.StreamHandler(log_capture)
    handler.setLevel(logging.DEBUG)
    
    # Get the logger from gpu_server
    logger = logging.getLogger('gpu_server')
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    
    # Test case: Some GPUs exist, some don't
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,2,5,7'
    
    from gpu_server import GPUMonitor
    monitor = GPUMonitor()
    
    # Mock GPU data with only GPUs 0, 1, 2, 3 available
    mock_gpu_data = {
        "gpu_count": 4,
        "gpus": [
            {"gpu_id": 0, "name": "Tesla V100", "memory": {"total_mb": 16384, "used_mb": 1024, "free_mb": 15360, "usage_percent": 6.25}, "utilization_percent": 15},
            {"gpu_id": 1, "name": "Tesla V100", "memory": {"total_mb": 16384, "used_mb": 2048, "free_mb": 14336, "usage_percent": 12.5}, "utilization_percent": 25},
            {"gpu_id": 2, "name": "Tesla V100", "memory": {"total_mb": 16384, "used_mb": 4096, "free_mb": 12288, "usage_percent": 25.0}, "utilization_percent": 50},
            {"gpu_id": 3, "name": "Tesla V100", "memory": {"total_mb": 16384, "used_mb": 8192, "free_mb": 8192, "usage_percent": 50.0}, "utilization_percent": 75}
        ],
        "timestamp": "2024-01-01T12:00:00"
    }
    
    print("Original GPU data:")
    print(f"  Total GPUs: {mock_gpu_data['gpu_count']}")
    print(f"  Available GPU IDs: {[gpu['gpu_id'] for gpu in mock_gpu_data['gpus']]}")
    print(f"  CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")
    print()
    
    # Apply filtering
    filtered_data = monitor._filter_gpu_data(mock_gpu_data)
    
    print("Filtered GPU data:")
    print(f"  Visible GPUs: {filtered_data['gpu_count']}")
    print(f"  Visible GPU IDs: {[gpu['gpu_id'] for gpu in filtered_data['gpus']]}")
    print()
    
    # Show the logs
    log_output = log_capture.getvalue()
    print("Generated logs:")
    print("-" * 40)
    for line in log_output.strip().split('\n'):
        if line.strip():
            print(line)
    print("-" * 40)
    
    # Verify the filtering worked correctly
    expected_visible_gpus = [0, 2]  # Only these exist in the mock data
    actual_visible_gpus = [gpu['gpu_id'] for gpu in filtered_data['gpus']]
    
    print(f"\nVerification:")
    print(f"  Expected visible GPUs: {expected_visible_gpus}")
    print(f"  Actual visible GPUs: {actual_visible_gpus}")
    print(f"  Match: {expected_visible_gpus == actual_visible_gpus}")
    
    # Check that the logs contain the expected warnings
    expected_warnings = [
        "GPUs [5, 7] specified in CUDA_VISIBLE_DEVICES do not exist on this system",
        "Available GPU IDs on system: [0, 1, 2, 3]",
        "Continuing with 2 valid GPUs: [0, 2]",
        "GPU filtering applied: 4 total system GPUs -> 2 visible GPUs"
    ]
    
    print(f"\nLog verification:")
    for warning in expected_warnings:
        if warning in log_output:
            print(f"  ✓ Found: {warning}")
        else:
            print(f"  ✗ Missing: {warning}")
    
    logger.removeHandler(handler)
    
    print("\n" + "=" * 60)
    print("GPU filtering logs test completed!")

if __name__ == "__main__":
    test_filtering_logs()