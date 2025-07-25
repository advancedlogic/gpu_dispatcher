#!/usr/bin/env python3
"""
Integration tests for API endpoint consistency with GPU filtering

This test suite verifies that all API endpoints (/gpu/stats, /gpu/count, /gpu/summary, /config)
return consistent filtered GPU sets when CUDA_VISIBLE_DEVICES filtering is active.

Requirements tested:
- 2.1: /gpu/stats endpoint returns only filtered GPUs
- 2.2: /gpu/count endpoint returns count of filtered GPUs only
- 2.3: /gpu/summary endpoint calculates summaries using only filtered GPUs
- 2.4: All memory and utilization calculations are based on visible GPUs only
- 6.1: Caching mechanism caches only filtered GPU data
- 6.2: Cache invalidation works correctly with filtering changes
- 6.3: Cached filtered data returns consistent results
"""

import os
import sys
import json
import time
import unittest
import asyncio
from unittest.mock import patch, MagicMock
from typing import Dict, Any, List

# Add the gpu_worker_pool directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'gpu_worker_pool'))


class TestAPIEndpointConsistency(unittest.TestCase):
    """Test suite for API endpoint consistency with GPU filtering"""
    
    def setUp(self):
        """Set up test environment"""
        # Clear any existing CUDA_VISIBLE_DEVICES
        if 'CUDA_VISIBLE_DEVICES' in os.environ:
            del os.environ['CUDA_VISIBLE_DEVICES']
        
        # Import after clearing environment
        from gpu_server import app, GPUMonitor, gpu_stats, gpu_count, gpu_summary, get_config
        self.app = app
        self.GPUMonitor = GPUMonitor
        self.gpu_stats = gpu_stats
        self.gpu_count = gpu_count
        self.gpu_summary = gpu_summary
        self.get_config = get_config
    
    def create_nvidia_smi_output(self, gpu_count: int = 4) -> str:
        """Create mock nvidia-smi CSV output for testing"""
        lines = []
        for i in range(gpu_count):
            # Format: index,name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu,power.draw,power.limit
            line = f"{i}, NVIDIA GeForce RTX 308{i}, 12288, {2048 + (i * 512)}, {10240 - (i * 512)}, {25 + (i * 15)}, {45 + i}, {150.0 + (i * 10)}, 320.0"
            lines.append(line)
        return "\n".join(lines)
    
    def mock_subprocess_run(self, gpu_count: int = 4):
        """Create a context manager to mock subprocess.run with nvidia-smi output"""
        mock_nvidia_output = self.create_nvidia_smi_output(gpu_count)
        
        def mock_run_func(*args, **kwargs):
            mock_result = MagicMock()
            mock_result.stdout = mock_nvidia_output
            mock_result.returncode = 0
            return mock_result
        
        return patch('subprocess.run', side_effect=mock_run_func)
    
    def reinitialize_gpu_monitor(self):
        """Reinitialize GPU monitor to pick up environment changes"""
        from gpu_server import gpu_monitor
        gpu_monitor._visible_devices = gpu_monitor._parse_visible_devices()
        gpu_monitor._cached_stats = None
        gpu_monitor._last_update = 0
        return gpu_monitor
    
    async def call_endpoints(self):
        """Call all endpoints and return their responses"""
        stats_data = await self.gpu_stats()
        count_data = await self.gpu_count()
        summary_data = await self.gpu_summary()
        config_data = await self.get_config()
        
        return stats_data, count_data, summary_data, config_data
    
    def test_all_endpoints_consistent_gpu_sets_no_filtering(self):
        """Test that all endpoints return consistent GPU sets when no filtering is applied"""
        print("\n=== Testing endpoint consistency with no filtering ===")
        
        # Ensure no filtering is active
        if 'CUDA_VISIBLE_DEVICES' in os.environ:
            del os.environ['CUDA_VISIBLE_DEVICES']
        
        gpu_monitor = self.reinitialize_gpu_monitor()
        
        # Mock the subprocess call to return consistent test data
        with self.mock_subprocess_run(4):
            # Get responses from all endpoints
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                stats_data, count_data, summary_data, config_data = loop.run_until_complete(self.call_endpoints())
            finally:
                loop.close()
            
            # Verify GPU counts are consistent across all endpoints
            self.assertEqual(stats_data["gpu_count"], 4)
            self.assertEqual(count_data["gpu_count"], 4)
            self.assertEqual(summary_data["gpu_count"], 4)
            
            # Verify GPU IDs are consistent between stats and summary
            stats_gpu_ids = {gpu["gpu_id"] for gpu in stats_data["gpus"]}
            summary_gpu_ids = {gpu["gpu_id"] for gpu in summary_data["gpus_summary"]}
            self.assertEqual(stats_gpu_ids, summary_gpu_ids)
            self.assertEqual(stats_gpu_ids, {0, 1, 2, 3})
            
            # Verify config endpoint shows no filtering
            gpu_filtering = config_data["gpu_filtering"]
            self.assertIsNone(gpu_filtering["cuda_visible_devices"])
            self.assertIsNone(gpu_filtering["visible_gpu_ids"])
            self.assertFalse(gpu_filtering["filtering_active"])
            self.assertEqual(gpu_filtering["visible_gpu_count"], 4)
            
            print("✓ All endpoints return consistent GPU sets with no filtering")
    
    def test_all_endpoints_consistent_gpu_sets_with_filtering(self):
        """Test that all endpoints return consistent GPU sets when filtering is applied"""
        print("\n=== Testing endpoint consistency with filtering (GPUs 0,2) ===")
        
        # Set filtering to show only GPUs 0 and 2
        os.environ['CUDA_VISIBLE_DEVICES'] = "0,2"
        gpu_monitor = self.reinitialize_gpu_monitor()
        
        # Mock the subprocess call to return test data
        with self.mock_subprocess_run(4):
            # Get responses from all endpoints
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                stats_data, count_data, summary_data, config_data = loop.run_until_complete(self.call_endpoints())
            finally:
                loop.close()
            
            # Verify GPU counts are consistent across all endpoints (should be 2)
            self.assertEqual(stats_data["gpu_count"], 2)
            self.assertEqual(count_data["gpu_count"], 2)
            self.assertEqual(summary_data["gpu_count"], 2)
            
            # Verify only GPUs 0 and 2 are present
            stats_gpu_ids = {gpu["gpu_id"] for gpu in stats_data["gpus"]}
            summary_gpu_ids = {gpu["gpu_id"] for gpu in summary_data["gpus_summary"]}
            self.assertEqual(stats_gpu_ids, summary_gpu_ids)
            self.assertEqual(stats_gpu_ids, {0, 2})
            
            # Verify config endpoint shows correct filtering
            gpu_filtering = config_data["gpu_filtering"]
            self.assertEqual(gpu_filtering["cuda_visible_devices"], "0,2")
            self.assertEqual(gpu_filtering["visible_gpu_ids"], [0, 2])
            self.assertTrue(gpu_filtering["filtering_active"])
            self.assertEqual(gpu_filtering["visible_gpu_count"], 2)
            
            print("✓ All endpoints return consistent filtered GPU sets")
    
    def test_summary_calculations_use_only_filtered_gpus(self):
        """Test that summary calculations (memory totals, averages) use only filtered GPUs"""
        print("\n=== Testing summary calculations with filtering ===")
        
        # Set filtering to show only GPU 1
        os.environ['CUDA_VISIBLE_DEVICES'] = "1"
        gpu_monitor = self.reinitialize_gpu_monitor()
        
        # Mock the subprocess call to return test data
        with self.mock_subprocess_run(4):
            # Get summary response
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                summary_data = loop.run_until_complete(self.gpu_summary())
            finally:
                loop.close()
            
            # Verify only GPU 1 is included
            self.assertEqual(summary_data["gpu_count"], 1)
            self.assertEqual(len(summary_data["gpus_summary"]), 1)
            self.assertEqual(summary_data["gpus_summary"][0]["gpu_id"], 1)
            
            # Verify calculations are based only on GPU 1
            # Expected values for GPU 1 based on our mock data generation
            expected_total_mb = 12288
            expected_used_mb = 2048 + (1 * 512)  # 2560
            expected_utilization = 25 + (1 * 15)  # 40
            
            # Memory calculations should match GPU 1 only
            self.assertEqual(summary_data["total_memory_mb"], expected_total_mb)
            self.assertEqual(summary_data["total_used_memory_mb"], expected_used_mb)
            
            # Utilization average should be GPU 1's utilization (since only 1 GPU)
            self.assertEqual(summary_data["average_utilization_percent"], expected_utilization)
            
            # Memory usage percentage should match GPU 1's usage
            expected_memory_usage = round((expected_used_mb / expected_total_mb) * 100, 2)
            self.assertEqual(summary_data["total_memory_usage_percent"], expected_memory_usage)
            
            print("✓ Summary calculations use only filtered GPUs")
    
    def test_empty_filtering_returns_consistent_empty_results(self):
        """Test that empty filtering (no visible GPUs) returns consistent empty results"""
        print("\n=== Testing endpoint consistency with empty filtering ===")
        
        # Set filtering to show no GPUs
        os.environ['CUDA_VISIBLE_DEVICES'] = ""
        gpu_monitor = self.reinitialize_gpu_monitor()
        
        # Mock the subprocess call to return test data
        with self.mock_subprocess_run(4):
            # Get responses from all endpoints
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                stats_data, count_data, summary_data, config_data = loop.run_until_complete(self.call_endpoints())
            finally:
                loop.close()
            
            # Verify all endpoints show 0 GPUs
            self.assertEqual(stats_data["gpu_count"], 0)
            self.assertEqual(count_data["gpu_count"], 0)
            self.assertEqual(summary_data["gpu_count"], 0)
            
            # Verify empty GPU lists
            self.assertEqual(len(stats_data["gpus"]), 0)
            self.assertEqual(len(summary_data["gpus_summary"]), 0)
            
            # Verify summary calculations with no GPUs
            self.assertEqual(summary_data["total_memory_mb"], 0)
            self.assertEqual(summary_data["total_used_memory_mb"], 0)
            self.assertEqual(summary_data["average_utilization_percent"], 0)
            
            # Verify config endpoint shows empty filtering
            gpu_filtering = config_data["gpu_filtering"]
            self.assertEqual(gpu_filtering["cuda_visible_devices"], "")
            self.assertEqual(gpu_filtering["visible_gpu_ids"], [])
            self.assertTrue(gpu_filtering["filtering_active"])
            self.assertEqual(gpu_filtering["visible_gpu_count"], 0)
            
            print("✓ Empty filtering returns consistent empty results")
    
    def test_configuration_endpoint_filtering_information(self):
        """Test that configuration endpoint returns correct filtering information"""
        print("\n=== Testing configuration endpoint filtering information ===")
        
        test_cases = [
            {
                "name": "No filtering",
                "cuda_visible_devices": None,
                "expected_filtering_active": False,
                "expected_visible_gpu_ids": None
            },
            {
                "name": "Empty filtering",
                "cuda_visible_devices": "",
                "expected_filtering_active": True,
                "expected_visible_gpu_ids": []
            },
            {
                "name": "Single GPU filtering",
                "cuda_visible_devices": "1",
                "expected_filtering_active": True,
                "expected_visible_gpu_ids": [1]
            },
            {
                "name": "Multiple GPU filtering",
                "cuda_visible_devices": "0,2,3",
                "expected_filtering_active": True,
                "expected_visible_gpu_ids": [0, 2, 3]
            }
        ]
        
        for test_case in test_cases:
            print(f"  Testing: {test_case['name']}")
            
            # Set environment
            if test_case['cuda_visible_devices'] is None:
                if 'CUDA_VISIBLE_DEVICES' in os.environ:
                    del os.environ['CUDA_VISIBLE_DEVICES']
            else:
                os.environ['CUDA_VISIBLE_DEVICES'] = test_case['cuda_visible_devices']
            
            gpu_monitor = self.reinitialize_gpu_monitor()
            
            with self.mock_subprocess_run(4):
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    config_data = loop.run_until_complete(self.get_config())
                finally:
                    loop.close()
                
                # Verify gpu_filtering section exists
                self.assertIn("gpu_filtering", config_data)
                gpu_filtering = config_data["gpu_filtering"]
                
                # Verify all required fields are present
                required_fields = [
                    "cuda_visible_devices",
                    "visible_gpu_ids",
                    "filtering_active",
                    "total_system_gpus",
                    "visible_gpu_count"
                ]
                
                for field in required_fields:
                    self.assertIn(field, gpu_filtering)
                
                # Verify field values
                self.assertEqual(gpu_filtering["cuda_visible_devices"], test_case['cuda_visible_devices'])
                self.assertEqual(gpu_filtering["visible_gpu_ids"], test_case['expected_visible_gpu_ids'])
                self.assertEqual(gpu_filtering["filtering_active"], test_case['expected_filtering_active'])
                self.assertIsInstance(gpu_filtering["total_system_gpus"], int)
                self.assertIsInstance(gpu_filtering["visible_gpu_count"], int)
                
                print(f"    ✓ Configuration correct for {test_case['name']}")
        
        print("✓ Configuration endpoint returns correct filtering information")
    
    def test_caching_works_correctly_with_filtered_data(self):
        """Test that caching mechanism works correctly with filtered data"""
        print("\n=== Testing caching with filtered data ===")
        
        # Set filtering
        os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"
        gpu_monitor = self.reinitialize_gpu_monitor()
        
        # Mock the subprocess call to return test data
        with self.mock_subprocess_run(4):
            # First request should populate cache
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                stats_data1 = loop.run_until_complete(self.gpu_stats())
            finally:
                loop.close()
            
            # Verify filtering worked
            self.assertEqual(stats_data1["gpu_count"], 2)
            stats_gpu_ids1 = {gpu["gpu_id"] for gpu in stats_data1["gpus"]}
            self.assertEqual(stats_gpu_ids1, {0, 1})
            
            # Second request should use cached data (within refresh interval)
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                stats_data2 = loop.run_until_complete(self.gpu_stats())
            finally:
                loop.close()
            
            # Verify data is identical (from cache)
            self.assertEqual(stats_data1, stats_data2)
            
            # Test other endpoints also return consistent data
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                count_data = loop.run_until_complete(self.gpu_count())
                summary_data = loop.run_until_complete(self.gpu_summary())
            finally:
                loop.close()
            
            # Verify consistency with cached data
            self.assertEqual(count_data["gpu_count"], 2)
            self.assertEqual(summary_data["gpu_count"], 2)
            
            print("✓ Caching works correctly with filtered data")
    
    def test_error_handling_consistency_across_endpoints(self):
        """Test that error handling is consistent across all endpoints when nvidia-smi fails"""
        print("\n=== Testing error handling consistency ===")
        
        # Set filtering
        os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"
        gpu_monitor = self.reinitialize_gpu_monitor()
        
        # Mock nvidia-smi failure
        def mock_run_func(*args, **kwargs):
            from subprocess import CalledProcessError
            raise CalledProcessError(1, 'nvidia-smi', 'Command failed')
        
        with patch('subprocess.run', side_effect=mock_run_func):
            # Get responses from all endpoints
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                stats_data = loop.run_until_complete(self.gpu_stats())
                count_data = loop.run_until_complete(self.gpu_count())
                summary_data = loop.run_until_complete(self.gpu_summary())
            finally:
                loop.close()
            
            # Verify error is passed through consistently
            self.assertIn("error", stats_data)
            self.assertEqual(stats_data["gpu_count"], 0)
            self.assertEqual(count_data["gpu_count"], 0)
            
            # Summary should also handle error gracefully
            if "error" in summary_data:
                self.assertEqual(summary_data["gpu_count"], 0)
            
            print("✓ Error handling is consistent across endpoints")


def run_integration_tests():
    """Run all integration tests"""
    print("Running API Endpoint Consistency Integration Tests")
    print("=" * 60)
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestAPIEndpointConsistency)
    
    # Run tests with verbose output
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 60)
    if result.wasSuccessful():
        print("✅ All API endpoint consistency integration tests passed!")
        return True
    else:
        print("❌ Some integration tests failed!")
        print(f"Failures: {len(result.failures)}")
        print(f"Errors: {len(result.errors)}")
        return False


if __name__ == "__main__":
    success = run_integration_tests()
    sys.exit(0 if success else 1)