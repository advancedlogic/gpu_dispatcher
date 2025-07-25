#!/usr/bin/env python3
"""
Test GPU filtering functionality for API endpoints

This test verifies that the GPU filtering logic works correctly and that
all API endpoints return consistent filtered GPU sets.

Requirements tested:
- 2.1: /gpu/stats endpoint returns only filtered GPUs
- 2.2: /gpu/count endpoint returns count of filtered GPUs only  
- 2.3: /gpu/summary endpoint calculates summaries using only filtered GPUs
- 2.4: All memory and utilization calculations are based on visible GPUs only
"""

import os
import sys
from unittest.mock import patch

# Add the gpu_worker_pool directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'gpu_worker_pool'))

from gpu_server import GPUMonitor

# Mock GPU data for testing
MOCK_GPU_DATA = {
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

class TestGPUFiltering:
    """Test class for GPU filtering functionality"""
    
    def test_filter_gpu_data_with_visible_devices(self):
        """Test filtering GPU data with specific visible devices"""
        # Create GPU monitor and set visible devices manually
        monitor = GPUMonitor()
        monitor._visible_devices = [0, 2]  # Only GPUs 0 and 2 should be visible
        
        # Apply filtering to mock data
        filtered_data = monitor._filter_gpu_data(MOCK_GPU_DATA)
        
        # Verify filtering results
        assert filtered_data["gpu_count"] == 2
        assert len(filtered_data["gpus"]) == 2
        
        # Verify only GPUs 0 and 2 are present
        gpu_ids = {gpu["gpu_id"] for gpu in filtered_data["gpus"]}
        assert gpu_ids == {0, 2}
        
        # Verify GPU data integrity
        gpu_0 = next(gpu for gpu in filtered_data["gpus"] if gpu["gpu_id"] == 0)
        gpu_2 = next(gpu for gpu in filtered_data["gpus"] if gpu["gpu_id"] == 2)
        
        assert gpu_0["name"] == "NVIDIA GeForce RTX 3080"
        assert gpu_2["name"] == "NVIDIA GeForce RTX 4090"
        
        print("✓ GPU data filtering with visible devices works correctly")
    
    def test_filter_gpu_data_no_filtering(self):
        """Test that no filtering returns all GPUs"""
        # Create GPU monitor with no filtering
        monitor = GPUMonitor()
        monitor._visible_devices = None  # No filtering
        
        # Apply filtering to mock data
        filtered_data = monitor._filter_gpu_data(MOCK_GPU_DATA)
        
        # Verify no filtering occurred
        assert filtered_data["gpu_count"] == 4
        assert len(filtered_data["gpus"]) == 4
        
        # Verify all GPU IDs are present
        gpu_ids = {gpu["gpu_id"] for gpu in filtered_data["gpus"]}
        assert gpu_ids == {0, 1, 2, 3}
        
        print("✓ No filtering returns all GPUs correctly")
    
    def test_filter_gpu_data_empty_visible_devices(self):
        """Test filtering with empty visible devices list"""
        # Create GPU monitor with empty visible devices
        monitor = GPUMonitor()
        monitor._visible_devices = []  # No GPUs visible
        
        # Apply filtering to mock data
        filtered_data = monitor._filter_gpu_data(MOCK_GPU_DATA)
        
        # Verify no GPUs are visible
        assert filtered_data["gpu_count"] == 0
        assert len(filtered_data["gpus"]) == 0
        
        print("✓ Empty visible devices filtering works correctly")
    
    def test_filter_gpu_data_nonexistent_gpus(self):
        """Test filtering with non-existent GPU IDs"""
        # Create GPU monitor with non-existent GPU IDs
        monitor = GPUMonitor()
        monitor._visible_devices = [0, 5]  # GPU 5 doesn't exist
        
        # Apply filtering to mock data
        filtered_data = monitor._filter_gpu_data(MOCK_GPU_DATA)
        
        # Should only show GPU 0 (GPU 5 doesn't exist)
        assert filtered_data["gpu_count"] == 1
        assert len(filtered_data["gpus"]) == 1
        assert filtered_data["gpus"][0]["gpu_id"] == 0
        
        print("✓ Non-existent GPU filtering works correctly")
    
    def test_filter_gpu_data_error_passthrough(self):
        """Test that error responses pass through filtering unchanged"""
        # Create GPU monitor with filtering
        monitor = GPUMonitor()
        monitor._visible_devices = [0, 2]
        
        # Create error data
        error_data = {
            "error": "nvidia-smi failed",
            "gpu_count": 0,
            "gpus": []
        }
        
        # Apply filtering to error data
        filtered_data = monitor._filter_gpu_data(error_data)
        
        # Verify error data passes through unchanged
        assert filtered_data == error_data
        assert "error" in filtered_data
        
        print("✓ Error responses pass through filtering unchanged")
    
    def test_api_endpoint_consistency(self):
        """Test that API endpoint simulations return consistent filtered data"""
        # Create GPU monitor with filtering
        monitor = GPUMonitor()
        monitor._visible_devices = [1, 3]  # Only GPUs 1 and 3 visible
        
        # Mock the _fetch_gpu_stats method to return our test data
        def mock_fetch_gpu_stats():
            return monitor._filter_gpu_data(MOCK_GPU_DATA)
        
        monitor._fetch_gpu_stats = mock_fetch_gpu_stats
        
        # Simulate API endpoints
        stats_data = monitor.get_gpu_stats()
        
        # Simulate /gpu/count endpoint
        count_data = {
            "gpu_count": stats_data.get("gpu_count", 0),
            "timestamp": stats_data.get("timestamp")
        }
        
        # Simulate /gpu/summary endpoint
        summary_data = self._simulate_gpu_summary(stats_data)
        
        # Verify consistent GPU count across all endpoints
        assert stats_data["gpu_count"] == 2
        assert count_data["gpu_count"] == 2
        assert summary_data["gpu_count"] == 2
        
        # Verify only GPUs 1 and 3 are present
        gpu_ids_in_stats = {gpu["gpu_id"] for gpu in stats_data["gpus"]}
        assert gpu_ids_in_stats == {1, 3}
        
        gpu_ids_in_summary = {gpu["gpu_id"] for gpu in summary_data["gpus_summary"]}
        assert gpu_ids_in_summary == {1, 3}
        
        # Verify memory calculations use only filtered GPUs
        # GPU 1: 24576 MB total, 4096 MB used, 30% utilization
        # GPU 3: 32768 MB total, 8192 MB used, 80% utilization
        expected_total_memory = 24576 + 32768  # 57344 MB
        expected_used_memory = 4096 + 8192     # 12288 MB
        expected_avg_utilization = (30 + 80) / 2  # 55%
        
        assert summary_data["total_memory_mb"] == expected_total_memory
        assert summary_data["total_used_memory_mb"] == expected_used_memory
        assert summary_data["average_utilization_percent"] == expected_avg_utilization
        
        print("✓ API endpoint consistency with filtering works correctly")
    
    def _simulate_gpu_summary(self, stats):
        """Simulate /gpu/summary endpoint behavior"""
        if "error" in stats:
            return stats
        
        summary = {
            "gpu_count": stats["gpu_count"],
            "total_memory_mb": 0,
            "total_used_memory_mb": 0,
            "average_utilization_percent": 0,
            "gpus_summary": []
        }
        
        if stats["gpus"]:
            total_util = 0
            for gpu in stats["gpus"]:
                summary["total_memory_mb"] += gpu["memory"]["total_mb"]
                summary["total_used_memory_mb"] += gpu["memory"]["used_mb"]
                total_util += gpu["utilization_percent"]
                
                summary["gpus_summary"].append({
                    "gpu_id": gpu["gpu_id"],
                    "name": gpu["name"],
                    "memory_usage_percent": gpu["memory"]["usage_percent"],
                    "utilization_percent": gpu["utilization_percent"]
                })
            
            summary["average_utilization_percent"] = round(total_util / len(stats["gpus"]), 2)
            summary["total_memory_usage_percent"] = round(
                (summary["total_used_memory_mb"] / summary["total_memory_mb"]) * 100, 2
            ) if summary["total_memory_mb"] > 0 else 0
        
        summary["timestamp"] = stats.get("timestamp")
        return summary
    
    @patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '0,2'})
    def test_parse_visible_devices_comma_separated(self):
        """Test parsing comma-separated CUDA_VISIBLE_DEVICES"""
        monitor = GPUMonitor()
        visible_devices = monitor._parse_visible_devices()
        
        assert visible_devices == [0, 2]
        print("✓ Comma-separated CUDA_VISIBLE_DEVICES parsing works correctly")
    
    @patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '0 1 2'})
    def test_parse_visible_devices_space_separated(self):
        """Test parsing space-separated CUDA_VISIBLE_DEVICES"""
        monitor = GPUMonitor()
        visible_devices = monitor._parse_visible_devices()
        
        assert visible_devices == [0, 1, 2]
        print("✓ Space-separated CUDA_VISIBLE_DEVICES parsing works correctly")
    
    @patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': ''})
    def test_parse_visible_devices_empty(self):
        """Test parsing empty CUDA_VISIBLE_DEVICES"""
        monitor = GPUMonitor()
        visible_devices = monitor._parse_visible_devices()
        
        assert visible_devices == []
        print("✓ Empty CUDA_VISIBLE_DEVICES parsing works correctly")
    
    @patch.dict(os.environ, {}, clear=True)
    def test_parse_visible_devices_not_set(self):
        """Test parsing when CUDA_VISIBLE_DEVICES is not set"""
        monitor = GPUMonitor()
        visible_devices = monitor._parse_visible_devices()
        
        assert visible_devices is None
        print("✓ Unset CUDA_VISIBLE_DEVICES parsing works correctly")

def run_tests():
    """Run all tests"""
    print("Testing GPU filtering functionality...")
    print("=" * 60)
    
    test_instance = TestGPUFiltering()
    
    try:
        test_instance.test_filter_gpu_data_with_visible_devices()
        test_instance.test_filter_gpu_data_no_filtering()
        test_instance.test_filter_gpu_data_empty_visible_devices()
        test_instance.test_filter_gpu_data_nonexistent_gpus()
        test_instance.test_filter_gpu_data_error_passthrough()
        test_instance.test_api_endpoint_consistency()
        test_instance.test_parse_visible_devices_comma_separated()
        test_instance.test_parse_visible_devices_space_separated()
        test_instance.test_parse_visible_devices_empty()
        test_instance.test_parse_visible_devices_not_set()
        
        print("=" * 60)
        print("✅ All GPU filtering tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)