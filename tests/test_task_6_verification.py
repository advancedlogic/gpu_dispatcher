#!/usr/bin/env python3
"""
Comprehensive verification test for Task 6: Add comprehensive error handling and logging
This test verifies all the requirements specified in the task details.
"""

import os
import sys
import logging
import io
from unittest.mock import patch

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

def test_requirement_3_1_invalid_gpu_indices():
    """Test requirement 3.1: Warning logs for invalid GPU indices in CUDA_VISIBLE_DEVICES"""
    print("Testing Requirement 3.1: Warning logs for invalid GPU indices...")
    
    log_capture, handler, logger = capture_logs()
    
    with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '0,abc,2,-1,xyz'}):
        from gpu_server import GPUMonitor
        monitor = GPUMonitor()
    
    log_output = log_capture.getvalue()
    
    # Verify specific warning messages for invalid indices
    checks = [
        ("Invalid GPU index 'abc' in CUDA_VISIBLE_DEVICES: must be a non-negative integer" in log_output, "Non-numeric value warning"),
        ("Invalid GPU index 'xyz' in CUDA_VISIBLE_DEVICES: must be a non-negative integer" in log_output, "Another non-numeric value warning"),
        ("Invalid GPU index '-1' in CUDA_VISIBLE_DEVICES: negative GPU indices are not allowed" in log_output, "Negative value warning"),
        ("CUDA_VISIBLE_DEVICES parsing found" in log_output and "invalid entries" in log_output, "Summary of invalid entries")
    ]
    
    for check, description in checks:
        print(f"  {'✓' if check else '✗'} {description}")
        assert check, f"Failed: {description}"
    
    logger.removeHandler(handler)
    print("  ✅ Requirement 3.1 verified\n")

def test_requirement_3_2_parsing_failure():
    """Test requirement 3.2: Error logging when CUDA_VISIBLE_DEVICES parsing fails completely"""
    print("Testing Requirement 3.2: Error logging when parsing fails completely...")
    
    log_capture, handler, logger = capture_logs()
    
    with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': 'abc,xyz,-1,-2'}):
        from gpu_server import GPUMonitor
        monitor = GPUMonitor()
    
    log_output = log_capture.getvalue()
    
    checks = [
        ("CUDA_VISIBLE_DEVICES parsing failed: no valid GPU indices found" in log_output, "Error message for complete parsing failure"),
        ("Falling back to showing all GPUs due to parsing errors" in log_output, "Fallback message"),
        (monitor._visible_devices is None, "Fallback to None (all GPUs)")
    ]
    
    for check, description in checks:
        print(f"  {'✓' if check else '✗'} {description}")
        assert check, f"Failed: {description}"
    
    logger.removeHandler(handler)
    print("  ✅ Requirement 3.2 verified\n")

def test_requirement_3_3_missing_gpus():
    """Test requirement 3.3: Log specific warnings when referenced GPUs don't exist on the system"""
    print("Testing Requirement 3.3: Warnings when referenced GPUs don't exist...")
    
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
        
        filtered_data = monitor._filter_gpu_data(mock_gpu_data)
    
    log_output = log_capture.getvalue()
    
    checks = [
        ("GPUs [5, 7] specified in CUDA_VISIBLE_DEVICES do not exist on this system" in log_output, "Specific missing GPU warning"),
        ("Available GPU IDs on system: [0, 1, 2]" in log_output, "Available GPUs information"),
        ("Continuing with 2 valid GPUs: [0, 2]" in log_output, "Valid GPUs continuation message"),
        (filtered_data["gpu_count"] == 2, "Correct filtering result")
    ]
    
    for check, description in checks:
        print(f"  {'✓' if check else '✗'} {description}")
        assert check, f"Failed: {description}"
    
    logger.removeHandler(handler)
    print("  ✅ Requirement 3.3 verified\n")

def test_requirement_3_4_graceful_fallback():
    """Test requirement 3.4: Ensure graceful fallback to all GPUs when parsing errors occur"""
    print("Testing Requirement 3.4: Graceful fallback to all GPUs...")
    
    log_capture, handler, logger = capture_logs()
    
    # Test critical parsing error
    with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '0,1,2'}):
        with patch('re.split', side_effect=Exception("Critical error")):
            from gpu_server import GPUMonitor
            monitor = GPUMonitor()
    
    log_output = log_capture.getvalue()
    
    checks = [
        ("Critical error parsing CUDA_VISIBLE_DEVICES" in log_output, "Critical error logged"),
        ("falling back to showing all GPUs" in log_output, "Fallback message"),
        (monitor._visible_devices is None, "Fallback to None (all GPUs)")
    ]
    
    for check, description in checks:
        print(f"  {'✓' if check else '✗'} {description}")
        assert check, f"Failed: {description}"
    
    logger.removeHandler(handler)
    print("  ✅ Requirement 3.4 verified\n")

def test_requirement_3_5_informative_startup_logs():
    """Test requirement 3.5: Add informative startup logs showing filtering configuration"""
    print("Testing Requirement 3.5: Informative startup logs...")
    
    log_capture, handler, logger = capture_logs()
    
    # Test different scenarios
    scenarios = [
        (None, "not set"),
        ("", "empty"),
        ("0,2", "specific GPUs")
    ]
    
    all_checks_passed = True
    
    for cuda_value, description in scenarios:
        log_capture.seek(0)
        log_capture.truncate(0)
        
        env_patch = {'CUDA_VISIBLE_DEVICES': cuda_value} if cuda_value is not None else {}
        if cuda_value is None and 'CUDA_VISIBLE_DEVICES' in os.environ:
            del os.environ['CUDA_VISIBLE_DEVICES']
        
        with patch.dict(os.environ, env_patch, clear=cuda_value is None):
            if 'gpu_server' in sys.modules:
                del sys.modules['gpu_server']
            
            from gpu_server import GPUMonitor
            monitor = GPUMonitor()
        
        log_output = log_capture.getvalue()
        
        check = f"CUDA_VISIBLE_DEVICES environment variable: {repr(cuda_value)}" in log_output
        print(f"  {'✓' if check else '✗'} Startup logging for {description}")
        if not check:
            all_checks_passed = False
    
    assert all_checks_passed, "Some startup logging checks failed"
    
    logger.removeHandler(handler)
    print("  ✅ Requirement 3.5 verified\n")

def test_requirement_4_1_4_2_startup_logging():
    """Test requirements 4.1 & 4.2: Server logs CUDA_VISIBLE_DEVICES value and visible GPU IDs at startup"""
    print("Testing Requirements 4.1 & 4.2: Startup logging of configuration...")
    
    log_capture, handler, logger = capture_logs()
    
    with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '0,2'}):
        from gpu_server import GPUMonitor
        monitor = GPUMonitor()
    
    log_output = log_capture.getvalue()
    
    checks = [
        ("CUDA_VISIBLE_DEVICES environment variable: '0,2'" in log_output, "CUDA_VISIBLE_DEVICES value logged"),
        ("Successfully parsed 2 visible GPU IDs from CUDA_VISIBLE_DEVICES: [0, 2]" in log_output, "Visible GPU IDs logged")
    ]
    
    for check, description in checks:
        print(f"  {'✓' if check else '✗'} {description}")
        assert check, f"Failed: {description}"
    
    logger.removeHandler(handler)
    print("  ✅ Requirements 4.1 & 4.2 verified\n")

def test_requirement_4_4_invalid_gpu_warnings():
    """Test requirement 4.4: Log specific warning messages for invalid GPU indices"""
    print("Testing Requirement 4.4: Specific warning messages for invalid GPU indices...")
    
    log_capture, handler, logger = capture_logs()
    
    with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '0,5,abc'}):
        from gpu_server import GPUMonitor
        monitor = GPUMonitor()
        
        # Mock filtering with missing GPU
        mock_gpu_data = {
            "gpu_count": 2,
            "gpus": [{"gpu_id": 0, "name": "GPU 0"}, {"gpu_id": 1, "name": "GPU 1"}],
            "timestamp": "2024-01-01T00:00:00"
        }
        
        filtered_data = monitor._filter_gpu_data(mock_gpu_data)
    
    log_output = log_capture.getvalue()
    
    checks = [
        ("Invalid GPU index 'abc' in CUDA_VISIBLE_DEVICES: must be a non-negative integer" in log_output, "Invalid format warning"),
        ("GPU 5 specified in CUDA_VISIBLE_DEVICES does not exist on this system" in log_output, "Missing GPU warning")
    ]
    
    for check, description in checks:
        print(f"  {'✓' if check else '✗'} {description}")
        assert check, f"Failed: {description}"
    
    logger.removeHandler(handler)
    print("  ✅ Requirement 4.4 verified\n")

def run_all_verification_tests():
    """Run all verification tests for Task 6"""
    print("=" * 80)
    print("TASK 6 VERIFICATION: Add comprehensive error handling and logging")
    print("=" * 80)
    print()
    
    try:
        test_requirement_3_1_invalid_gpu_indices()
        test_requirement_3_2_parsing_failure()
        test_requirement_3_3_missing_gpus()
        test_requirement_3_4_graceful_fallback()
        test_requirement_3_5_informative_startup_logs()
        test_requirement_4_1_4_2_startup_logging()
        test_requirement_4_4_invalid_gpu_warnings()
        
        print("=" * 80)
        print("🎉 ALL TASK 6 REQUIREMENTS VERIFIED SUCCESSFULLY!")
        print("=" * 80)
        print()
        print("Task 6 Implementation Summary:")
        print("✅ Warning logs for invalid GPU indices in CUDA_VISIBLE_DEVICES")
        print("✅ Error logging when CUDA_VISIBLE_DEVICES parsing fails completely")
        print("✅ Specific warnings when referenced GPUs don't exist on the system")
        print("✅ Graceful fallback to all GPUs when parsing errors occur")
        print("✅ Informative startup logs showing filtering configuration")
        print("✅ Server logs CUDA_VISIBLE_DEVICES value and visible GPU IDs at startup")
        print("✅ Specific warning messages for invalid GPU indices")
        print()
        return True
        
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_all_verification_tests()
    sys.exit(0 if success else 1)