#!/usr/bin/env python3
"""
Test script to verify comprehensive error handling and logging for CUDA_VISIBLE_DEVICES filtering.
This tests the implementation of task 6 from the GPU visible devices filtering spec.
"""

import os
import sys
import logging
import io
from contextlib import redirect_stderr
from unittest.mock import patch, MagicMock

# Add the gpu_worker_pool directory to the path
sys.path.insert(0, 'gpu_worker_pool')

def capture_logs():
    """Helper function to capture log output"""
    log_capture = io.StringIO()
    handler = logging.StreamHandler(log_capture)
    handler.setLevel(logging.DEBUG)
    
    # Get the logger from gpu_server
    logger = logging.getLogger('gpu_server')
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    
    return log_capture, handler, logger

def test_invalid_gpu_indices_logging():
    """Test warning logs for invalid GPU indices in CUDA_VISIBLE_DEVICES"""
    print("Testing invalid GPU indices logging...")
    
    log_capture, handler, logger = capture_logs()
    
    # Test case 1: Invalid non-numeric values
    with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '0,abc,2,xyz'}):
        from gpu_server import GPUMonitor
        monitor = GPUMonitor()
    
    log_output = log_capture.getvalue()
    
    # Check for specific warning messages
    assert "Invalid GPU index 'abc' in CUDA_VISIBLE_DEVICES: must be a non-negative integer" in log_output
    assert "Invalid GPU index 'xyz' in CUDA_VISIBLE_DEVICES: must be a non-negative integer" in log_output
    assert "CUDA_VISIBLE_DEVICES parsing found 2 invalid entries: ['abc', 'xyz']" in log_output
    assert "Continuing with 2 valid GPU indices: [0, 2]" in log_output
    
    logger.removeHandler(handler)
    print("✓ Invalid GPU indices logging test passed")

def test_negative_gpu_indices_logging():
    """Test warning logs for negative GPU indices"""
    print("Testing negative GPU indices logging...")
    
    log_capture, handler, logger = capture_logs()
    
    # Test case: Negative GPU indices
    with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '0,-1,2,-5'}):
        from gpu_server import GPUMonitor
        monitor = GPUMonitor()
    
    log_output = log_capture.getvalue()
    
    # Check for specific warning messages
    assert "Invalid GPU index '-1' in CUDA_VISIBLE_DEVICES: negative GPU indices are not allowed" in log_output
    assert "Invalid GPU index '-5' in CUDA_VISIBLE_DEVICES: negative GPU indices are not allowed" in log_output
    assert "CUDA_VISIBLE_DEVICES parsing found 2 invalid entries: ['-1', '-5']" in log_output
    assert "Continuing with 2 valid GPU indices: [0, 2]" in log_output
    
    logger.removeHandler(handler)
    print("✓ Negative GPU indices logging test passed")

def test_complete_parsing_failure_logging():
    """Test error logging when CUDA_VISIBLE_DEVICES parsing fails completely"""
    print("Testing complete parsing failure logging...")
    
    log_capture, handler, logger = capture_logs()
    
    # Test case: All invalid values
    with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': 'abc,xyz,-1,-2'}):
        from gpu_server import GPUMonitor
        monitor = GPUMonitor()
    
    log_output = log_capture.getvalue()
    
    # Check for error messages when no valid GPUs found
    assert "CUDA_VISIBLE_DEVICES parsing failed: no valid GPU indices found" in log_output
    assert "Falling back to showing all GPUs due to parsing errors" in log_output
    assert monitor._visible_devices is None  # Should fallback to None (all GPUs)
    
    logger.removeHandler(handler)
    print("✓ Complete parsing failure logging test passed")

def test_critical_parsing_error_logging():
    """Test error logging for critical parsing errors"""
    print("Testing critical parsing error logging...")
    
    log_capture, handler, logger = capture_logs()
    
    # Mock a critical error during parsing
    with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '0,1,2'}):
        with patch('re.split', side_effect=Exception("Critical parsing error")):
            from gpu_server import GPUMonitor
            monitor = GPUMonitor()
    
    log_output = log_capture.getvalue()
    
    # Check for critical error messages
    assert "Critical error parsing CUDA_VISIBLE_DEVICES '0,1,2': Exception: Critical parsing error" in log_output
    assert "This indicates a serious parsing failure - falling back to showing all GPUs" in log_output
    assert monitor._visible_devices is None  # Should fallback to None
    
    logger.removeHandler(handler)
    print("✓ Critical parsing error logging test passed")

def test_missing_gpu_warnings():
    """Test specific warnings when referenced GPUs don't exist on the system"""
    print("Testing missing GPU warnings...")
    
    log_capture, handler, logger = capture_logs()
    
    with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '0,2,5,7'}):
        from gpu_server import GPUMonitor
        monitor = GPUMonitor()
        
        # Mock GPU data with only GPUs 0, 1, 2 available
        mock_gpu_data = {
            "gpu_count": 3,
            "gpus": [
                {"gpu_id": 0, "name": "GPU 0"},
                {"gpu_id": 1, "name": "GPU 1"},
                {"gpu_id": 2, "name": "GPU 2"}
            ],
            "timestamp": "2024-01-01T00:00:00"
        }
        
        # Test filtering with missing GPUs
        filtered_data = monitor._filter_gpu_data(mock_gpu_data)
    
    log_output = log_capture.getvalue()
    
    # Check for specific missing GPU warnings
    assert "GPUs [5, 7] specified in CUDA_VISIBLE_DEVICES do not exist on this system" in log_output
    assert "Available GPU IDs on system: [0, 1, 2]" in log_output
    assert "Continuing with 2 valid GPUs: [0, 2]" in log_output
    assert "GPU filtering applied: 3 total system GPUs -> 2 visible GPUs" in log_output
    
    # Verify filtered data
    assert filtered_data["gpu_count"] == 2
    assert len(filtered_data["gpus"]) == 2
    assert filtered_data["gpus"][0]["gpu_id"] == 0
    assert filtered_data["gpus"][1]["gpu_id"] == 2
    
    logger.removeHandler(handler)
    print("✓ Missing GPU warnings test passed")

def test_all_gpus_missing_error():
    """Test error logging when all specified GPUs are missing"""
    print("Testing all GPUs missing error...")
    
    log_capture, handler, logger = capture_logs()
    
    with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '5,6,7'}):
        from gpu_server import GPUMonitor
        monitor = GPUMonitor()
        
        # Mock GPU data with only GPUs 0, 1, 2 available
        mock_gpu_data = {
            "gpu_count": 3,
            "gpus": [
                {"gpu_id": 0, "name": "GPU 0"},
                {"gpu_id": 1, "name": "GPU 1"},
                {"gpu_id": 2, "name": "GPU 2"}
            ],
            "timestamp": "2024-01-01T00:00:00"
        }
        
        # Test filtering with all missing GPUs
        filtered_data = monitor._filter_gpu_data(mock_gpu_data)
    
    log_output = log_capture.getvalue()
    
    # Check for error when all GPUs are missing
    assert "GPUs [5, 6, 7] specified in CUDA_VISIBLE_DEVICES do not exist on this system" in log_output
    assert "No valid GPUs found after filtering - all specified GPUs are missing from system" in log_output
    assert "Returning empty GPU list due to complete filtering mismatch" in log_output
    
    # Verify empty filtered data
    assert filtered_data["gpu_count"] == 0
    assert len(filtered_data["gpus"]) == 0
    
    logger.removeHandler(handler)
    print("✓ All GPUs missing error test passed")

def test_startup_logging_comprehensive():
    """Test comprehensive startup logging showing filtering configuration"""
    print("Testing comprehensive startup logging...")
    
    log_capture, handler, logger = capture_logs()
    
    # Test different CUDA_VISIBLE_DEVICES scenarios
    test_cases = [
        (None, "DISABLED", "not set"),
        ("", "ENABLED", "empty string"),
        ("0,2", "ENABLED", "specific GPU indices")
    ]
    
    for cuda_value, expected_status, expected_reason in test_cases:
        log_capture.seek(0)
        log_capture.truncate(0)
        
        env_patch = {'CUDA_VISIBLE_DEVICES': cuda_value} if cuda_value is not None else {}
        if cuda_value is None:
            env_patch = {}
            # Remove CUDA_VISIBLE_DEVICES if it exists
            if 'CUDA_VISIBLE_DEVICES' in os.environ:
                del os.environ['CUDA_VISIBLE_DEVICES']
        
        with patch.dict(os.environ, env_patch, clear=cuda_value is None):
            # Reload the module to test startup logging
            if 'gpu_server' in sys.modules:
                del sys.modules['gpu_server']
            
            from gpu_server import GPUMonitor
            monitor = GPUMonitor()
        
        log_output = log_capture.getvalue()
        
        # Check for comprehensive startup logging
        assert f"CUDA_VISIBLE_DEVICES environment variable: {repr(cuda_value)}" in log_output
        
        if cuda_value is None:
            assert "CUDA_VISIBLE_DEVICES not set, showing all GPUs" in log_output
        elif cuda_value == "":
            assert "CUDA_VISIBLE_DEVICES is empty, showing no GPUs" in log_output
        else:
            assert f"Successfully parsed" in log_output and "visible GPU IDs from CUDA_VISIBLE_DEVICES" in log_output
    
    logger.removeHandler(handler)
    print("✓ Comprehensive startup logging test passed")

def run_all_tests():
    """Run all error handling and logging tests"""
    print("Running comprehensive error handling and logging tests...")
    print("=" * 60)
    
    try:
        test_invalid_gpu_indices_logging()
        test_negative_gpu_indices_logging()
        test_complete_parsing_failure_logging()
        test_critical_parsing_error_logging()
        test_missing_gpu_warnings()
        test_all_gpus_missing_error()
        test_startup_logging_comprehensive()
        
        print("=" * 60)
        print("✅ All error handling and logging tests passed!")
        print("Task 6 implementation verified successfully.")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)