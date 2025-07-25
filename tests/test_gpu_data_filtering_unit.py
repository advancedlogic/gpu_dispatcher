#!/usr/bin/env python3
"""
Unit tests for GPU data filtering functionality

This test file focuses specifically on testing the _filter_gpu_data method
of the GPUMonitor class to ensure it handles all edge cases correctly.

Requirements tested:
- 3.1: Invalid GPU indices handling with warnings
- 3.3: Non-existent GPU IDs handling with warnings  
- 3.4: Graceful fallback behavior
"""

import os
import sys
import logging
from unittest.mock import patch, MagicMock
from io import StringIO

# Add the gpu_worker_pool directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'gpu_worker_pool'))

from gpu_server import GPUMonitor

# Mock GPU data for testing
MOCK_GPU_DATA_4_GPUS = {
    "gpu_count": 4,
    "gpus": [
        {
            "gpu_id": 0,
            "name": "NVIDIA GeForce RTX 3080",
            "memory": {"total_mb": 10240, "used_mb": 2048, "free_mb": 8192, "usage_percent": 20.0},
            "utilization_percent": 45,
            "temperature_c": 65,
            "power": {"draw_w": 220.5, "limit_w": 320.0}
        },
        {
            "gpu_id": 1,
            "name": "NVIDIA GeForce RTX 3090",
            "memory": {"total_mb": 24576, "used_mb": 4096, "free_mb": 20480, "usage_percent": 16.67},
            "utilization_percent": 30,
            "temperature_c": 70,
            "power": {"draw_w": 350.2, "limit_w": 400.0}
        },
        {
            "gpu_id": 2,
            "name": "NVIDIA GeForce RTX 4090",
            "memory": {"total_mb": 24576, "used_mb": 1024, "free_mb": 23552, "usage_percent": 4.17},
            "utilization_percent": 15,
            "temperature_c": 60,
            "power": {"draw_w": 180.0, "limit_w": 450.0}
        },
        {
            "gpu_id": 3,
            "name": "NVIDIA Tesla V100",
            "memory": {"total_mb": 32768, "used_mb": 8192, "free_mb": 24576, "usage_percent": 25.0},
            "utilization_percent": 80,
            "temperature_c": 75,
            "power": {"draw_w": 250.0, "limit_w": 300.0}
        }
    ],
    "timestamp": "2025-07-25T10:00:00.000000"
}

MOCK_GPU_DATA_2_GPUS = {
    "gpu_count": 2,
    "gpus": [
        {
            "gpu_id": 0,
            "name": "NVIDIA GeForce RTX 3080",
            "memory": {"total_mb": 10240, "used_mb": 2048, "free_mb": 8192, "usage_percent": 20.0},
            "utilization_percent": 45,
            "temperature_c": 65,
            "power": {"draw_w": 220.5, "limit_w": 320.0}
        },
        {
            "gpu_id": 1,
            "name": "NVIDIA GeForce RTX 3090",
            "memory": {"total_mb": 24576, "used_mb": 4096, "free_mb": 20480, "usage_percent": 16.67},
            "utilization_percent": 30,
            "temperature_c": 70,
            "power": {"draw_w": 350.2, "limit_w": 400.0}
        }
    ],
    "timestamp": "2025-07-25T10:00:00.000000"
}

ERROR_GPU_DATA = {
    "error": "nvidia-smi failed",
    "gpu_count": 0,
    "gpus": []
}


class TestGPUDataFilteringUnit:
    """Unit tests for GPU data filtering functionality"""
    
    def setup_method(self):
        """Setup method called before each test"""
        # Create a string buffer to capture log output
        self.log_capture = StringIO()
        self.log_handler = logging.StreamHandler(self.log_capture)
        self.log_handler.setLevel(logging.INFO)
        
        # Get the logger and add our handler
        self.logger = logging.getLogger('gpu_server')
        self.logger.addHandler(self.log_handler)
        self.logger.setLevel(logging.INFO)
    
    def teardown_method(self):
        """Teardown method called after each test"""
        # Remove our log handler
        self.logger.removeHandler(self.log_handler)
        self.log_capture.close()
    
    def get_log_output(self):
        """Get captured log output"""
        return self.log_capture.getvalue()
    
    def test_filtering_with_valid_gpu_ids_that_exist_on_system(self):
        """Test filtering with valid GPU IDs that exist on system"""
        # Create GPU monitor with specific visible devices
        monitor = GPUMonitor()
        monitor._visible_devices = [0, 2]  # Valid GPU IDs that exist in mock data
        
        # Apply filtering
        filtered_data = monitor._filter_gpu_data(MOCK_GPU_DATA_4_GPUS)
        
        # Verify filtering results
        assert filtered_data["gpu_count"] == 2, f"Expected 2 GPUs, got {filtered_data['gpu_count']}"
        assert len(filtered_data["gpus"]) == 2, f"Expected 2 GPU entries, got {len(filtered_data['gpus'])}"
        
        # Verify correct GPUs are included
        gpu_ids = {gpu["gpu_id"] for gpu in filtered_data["gpus"]}
        assert gpu_ids == {0, 2}, f"Expected GPU IDs {{0, 2}}, got {gpu_ids}"
        
        # Verify GPU data integrity is maintained
        gpu_0 = next(gpu for gpu in filtered_data["gpus"] if gpu["gpu_id"] == 0)
        gpu_2 = next(gpu for gpu in filtered_data["gpus"] if gpu["gpu_id"] == 2)
        
        assert gpu_0["name"] == "NVIDIA GeForce RTX 3080"
        assert gpu_0["memory"]["total_mb"] == 10240
        assert gpu_0["utilization_percent"] == 45
        
        assert gpu_2["name"] == "NVIDIA GeForce RTX 4090"
        assert gpu_2["memory"]["total_mb"] == 24576
        assert gpu_2["utilization_percent"] == 15
        
        # Verify timestamp is preserved
        assert filtered_data["timestamp"] == MOCK_GPU_DATA_4_GPUS["timestamp"]
        
        print("✓ Filtering with valid GPU IDs that exist on system works correctly")
    
    def test_filtering_with_nonexistent_gpu_ids_and_verify_warnings(self):
        """Test filtering with non-existent GPU IDs and verify warnings are logged"""
        # Create GPU monitor with mix of valid and invalid GPU IDs
        monitor = GPUMonitor()
        monitor._visible_devices = [0, 5, 7]  # GPU 0 exists, GPUs 5 and 7 don't exist
        
        # Apply filtering
        filtered_data = monitor._filter_gpu_data(MOCK_GPU_DATA_4_GPUS)
        
        # Verify only valid GPU is included
        assert filtered_data["gpu_count"] == 1, f"Expected 1 GPU, got {filtered_data['gpu_count']}"
        assert len(filtered_data["gpus"]) == 1, f"Expected 1 GPU entry, got {len(filtered_data['gpus'])}"
        assert filtered_data["gpus"][0]["gpu_id"] == 0
        
        # Verify warnings were logged for non-existent GPUs
        log_output = self.get_log_output()
        assert "GPUs [5, 7] specified in CUDA_VISIBLE_DEVICES do not exist on this system" in log_output
        assert "Available GPU IDs on system: [0, 1, 2, 3]" in log_output
        assert "Continuing with 1 valid GPUs: [0]" in log_output
        
        print("✓ Filtering with non-existent GPU IDs and warning verification works correctly")
    
    def test_filtering_with_all_nonexistent_gpu_ids(self):
        """Test filtering when all specified GPU IDs are non-existent"""
        # Create GPU monitor with all non-existent GPU IDs
        monitor = GPUMonitor()
        monitor._visible_devices = [5, 6, 7]  # None of these exist in mock data
        
        # Apply filtering
        filtered_data = monitor._filter_gpu_data(MOCK_GPU_DATA_4_GPUS)
        
        # Verify no GPUs are returned
        assert filtered_data["gpu_count"] == 0, f"Expected 0 GPUs, got {filtered_data['gpu_count']}"
        assert len(filtered_data["gpus"]) == 0, f"Expected 0 GPU entries, got {len(filtered_data['gpus'])}"
        
        # Verify appropriate error logging
        log_output = self.get_log_output()
        assert "GPUs [5, 6, 7] specified in CUDA_VISIBLE_DEVICES do not exist on this system" in log_output
        assert "No valid GPUs found after filtering - all specified GPUs are missing from system" in log_output
        assert "Returning empty GPU list due to complete filtering mismatch" in log_output
        
        print("✓ Filtering with all non-existent GPU IDs works correctly")
    
    def test_filtering_with_empty_visible_devices_list(self):
        """Test filtering with empty visible devices list (no GPUs shown)"""
        # Create GPU monitor with empty visible devices list
        monitor = GPUMonitor()
        monitor._visible_devices = []  # Empty list means no GPUs should be visible
        
        # Apply filtering
        filtered_data = monitor._filter_gpu_data(MOCK_GPU_DATA_4_GPUS)
        
        # Verify no GPUs are returned
        assert filtered_data["gpu_count"] == 0, f"Expected 0 GPUs, got {filtered_data['gpu_count']}"
        assert len(filtered_data["gpus"]) == 0, f"Expected 0 GPU entries, got {len(filtered_data['gpus'])}"
        
        # Verify timestamp is preserved
        assert filtered_data["timestamp"] == MOCK_GPU_DATA_4_GPUS["timestamp"]
        
        # Verify structure is correct
        assert "gpus" in filtered_data
        assert "gpu_count" in filtered_data
        assert "timestamp" in filtered_data
        
        print("✓ Filtering with empty visible devices list works correctly")
    
    def test_filtering_disabled_shows_all_gpus(self):
        """Test filtering disabled (None) shows all GPUs"""
        # Create GPU monitor with filtering disabled
        monitor = GPUMonitor()
        monitor._visible_devices = None  # None means no filtering, show all GPUs
        
        # Apply filtering
        filtered_data = monitor._filter_gpu_data(MOCK_GPU_DATA_4_GPUS)
        
        # Verify all GPUs are returned unchanged
        assert filtered_data["gpu_count"] == 4, f"Expected 4 GPUs, got {filtered_data['gpu_count']}"
        assert len(filtered_data["gpus"]) == 4, f"Expected 4 GPU entries, got {len(filtered_data['gpus'])}"
        
        # Verify all GPU IDs are present
        gpu_ids = {gpu["gpu_id"] for gpu in filtered_data["gpus"]}
        assert gpu_ids == {0, 1, 2, 3}, f"Expected all GPU IDs {{0, 1, 2, 3}}, got {gpu_ids}"
        
        # Verify data is identical to input
        assert filtered_data == MOCK_GPU_DATA_4_GPUS
        
        print("✓ Filtering disabled (None) shows all GPUs correctly")
    
    def test_filtering_with_error_responses_passes_through_errors(self):
        """Test filtering with error responses from nvidia-smi passes through errors"""
        # Create GPU monitor with filtering enabled
        monitor = GPUMonitor()
        monitor._visible_devices = [0, 1]  # Set up filtering
        
        # Apply filtering to error data
        filtered_data = monitor._filter_gpu_data(ERROR_GPU_DATA)
        
        # Verify error data passes through unchanged
        assert filtered_data == ERROR_GPU_DATA
        assert "error" in filtered_data
        assert filtered_data["error"] == "nvidia-smi failed"
        assert filtered_data["gpu_count"] == 0
        assert filtered_data["gpus"] == []
        
        print("✓ Filtering with error responses passes through errors correctly")
    
    def test_filtered_data_maintains_correct_structure_and_gpu_ids(self):
        """Test that filtered data maintains correct structure and original GPU IDs"""
        # Create GPU monitor with specific filtering
        monitor = GPUMonitor()
        monitor._visible_devices = [1, 3]  # Filter to GPUs 1 and 3
        
        # Apply filtering
        filtered_data = monitor._filter_gpu_data(MOCK_GPU_DATA_4_GPUS)
        
        # Verify structure is maintained
        assert "gpu_count" in filtered_data
        assert "gpus" in filtered_data
        assert "timestamp" in filtered_data
        assert isinstance(filtered_data["gpus"], list)
        assert isinstance(filtered_data["gpu_count"], int)
        
        # Verify GPU IDs are preserved (not remapped)
        gpu_ids = [gpu["gpu_id"] for gpu in filtered_data["gpus"]]
        assert gpu_ids == [1, 3], f"Expected GPU IDs [1, 3] in order, got {gpu_ids}"
        
        # Verify each GPU maintains its complete structure
        for gpu in filtered_data["gpus"]:
            assert "gpu_id" in gpu
            assert "name" in gpu
            assert "memory" in gpu
            assert "utilization_percent" in gpu
            assert "temperature_c" in gpu
            assert "power" in gpu
            
            # Verify memory structure
            memory = gpu["memory"]
            assert "total_mb" in memory
            assert "used_mb" in memory
            assert "free_mb" in memory
            assert "usage_percent" in memory
            
            # Verify power structure
            power = gpu["power"]
            assert "draw_w" in power
            assert "limit_w" in power
        
        # Verify specific GPU data integrity
        gpu_1 = next(gpu for gpu in filtered_data["gpus"] if gpu["gpu_id"] == 1)
        gpu_3 = next(gpu for gpu in filtered_data["gpus"] if gpu["gpu_id"] == 3)
        
        # Check GPU 1 data
        assert gpu_1["name"] == "NVIDIA GeForce RTX 3090"
        assert gpu_1["memory"]["total_mb"] == 24576
        assert gpu_1["utilization_percent"] == 30
        
        # Check GPU 3 data
        assert gpu_3["name"] == "NVIDIA Tesla V100"
        assert gpu_3["memory"]["total_mb"] == 32768
        assert gpu_3["utilization_percent"] == 80
        
        print("✓ Filtered data maintains correct structure and GPU IDs correctly")
    
    def test_filtering_with_single_nonexistent_gpu_warning(self):
        """Test filtering with single non-existent GPU produces appropriate warning"""
        # Create GPU monitor with one valid and one invalid GPU ID
        monitor = GPUMonitor()
        monitor._visible_devices = [0, 5]  # GPU 0 exists, GPU 5 doesn't exist
        
        # Apply filtering
        filtered_data = monitor._filter_gpu_data(MOCK_GPU_DATA_4_GPUS)
        
        # Verify only valid GPU is included
        assert filtered_data["gpu_count"] == 1
        assert filtered_data["gpus"][0]["gpu_id"] == 0
        
        # Verify single GPU warning format
        log_output = self.get_log_output()
        assert "GPU 5 specified in CUDA_VISIBLE_DEVICES does not exist on this system" in log_output
        assert "Available GPU IDs on system: [0, 1, 2, 3]" in log_output
        
        print("✓ Filtering with single non-existent GPU warning works correctly")
    
    def test_filtering_preserves_gpu_order_from_original_data(self):
        """Test that filtering preserves the order of GPUs from original data"""
        # Create GPU monitor with GPUs in reverse order
        monitor = GPUMonitor()
        monitor._visible_devices = [3, 1, 0]  # Specify in different order
        
        # Apply filtering
        filtered_data = monitor._filter_gpu_data(MOCK_GPU_DATA_4_GPUS)
        
        # Verify GPUs appear in original data order (0, 1, 3), not filter order (3, 1, 0)
        gpu_ids = [gpu["gpu_id"] for gpu in filtered_data["gpus"]]
        assert gpu_ids == [0, 1, 3], f"Expected GPU IDs [0, 1, 3] in original order, got {gpu_ids}"
        
        print("✓ Filtering preserves GPU order from original data correctly")
    
    def test_filtering_with_smaller_system_gpu_set(self):
        """Test filtering behavior with a smaller system GPU set"""
        # Create GPU monitor with filtering that includes some valid and some invalid GPUs
        monitor = GPUMonitor()
        monitor._visible_devices = [0, 1, 2, 3, 4]  # GPU 4 doesn't exist in 2-GPU system
        
        # Apply filtering to 2-GPU system data
        filtered_data = monitor._filter_gpu_data(MOCK_GPU_DATA_2_GPUS)
        
        # Verify only existing GPUs are included
        assert filtered_data["gpu_count"] == 2
        gpu_ids = {gpu["gpu_id"] for gpu in filtered_data["gpus"]}
        assert gpu_ids == {0, 1}
        
        # Verify warning for non-existent GPUs (plural form since multiple GPUs are missing)
        log_output = self.get_log_output()
        assert "GPUs [2, 3, 4] specified in CUDA_VISIBLE_DEVICES do not exist on this system" in log_output
        assert "Available GPU IDs on system: [0, 1]" in log_output
        
        print("✓ Filtering with smaller system GPU set works correctly")


def run_tests():
    """Run all unit tests for GPU data filtering"""
    print("Running unit tests for GPU data filtering...")
    print("=" * 70)
    
    test_instance = TestGPUDataFilteringUnit()
    
    test_methods = [
        'test_filtering_with_valid_gpu_ids_that_exist_on_system',
        'test_filtering_with_nonexistent_gpu_ids_and_verify_warnings',
        'test_filtering_with_all_nonexistent_gpu_ids',
        'test_filtering_with_empty_visible_devices_list',
        'test_filtering_disabled_shows_all_gpus',
        'test_filtering_with_error_responses_passes_through_errors',
        'test_filtered_data_maintains_correct_structure_and_gpu_ids',
        'test_filtering_with_single_nonexistent_gpu_warning',
        'test_filtering_preserves_gpu_order_from_original_data',
        'test_filtering_with_smaller_system_gpu_set'
    ]
    
    passed_tests = 0
    failed_tests = 0
    
    for test_method_name in test_methods:
        try:
            print(f"\nRunning {test_method_name}...")
            test_instance.setup_method()
            test_method = getattr(test_instance, test_method_name)
            test_method()
            test_instance.teardown_method()
            passed_tests += 1
        except Exception as e:
            print(f"❌ Test {test_method_name} failed: {e}")
            import traceback
            traceback.print_exc()
            failed_tests += 1
            test_instance.teardown_method()
    
    print("\n" + "=" * 70)
    print(f"Test Results: {passed_tests} passed, {failed_tests} failed")
    
    if failed_tests == 0:
        print("✅ All GPU data filtering unit tests passed!")
        return True
    else:
        print(f"❌ {failed_tests} tests failed!")
        return False


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)