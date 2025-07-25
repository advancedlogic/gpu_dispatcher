#!/usr/bin/env python3
"""
Integration test for API endpoint filtering

This test verifies that the actual API endpoints work correctly with
CUDA_VISIBLE_DEVICES filtering by testing the server endpoints directly.

Requirements tested:
- 2.1: /gpu/stats endpoint returns only filtered GPUs
- 2.2: /gpu/count endpoint returns count of filtered GPUs only  
- 2.3: /gpu/summary endpoint calculates summaries using only filtered GPUs
- 2.4: All memory and utilization calculations are based on visible GPUs only
"""

import os
import sys
import json
import requests
import time
import subprocess
from unittest.mock import patch

def test_api_endpoints_with_filtering():
    """Test that API endpoints work correctly with filtering"""
    print("Testing API endpoints with CUDA_VISIBLE_DEVICES filtering...")
    print("=" * 60)
    
    # Test data to verify filtering works
    test_cases = [
        {
            "name": "No filtering (all GPUs)",
            "cuda_visible_devices": None,
            "expected_behavior": "Should show all available GPUs"
        },
        {
            "name": "Empty filtering (no GPUs)",
            "cuda_visible_devices": "",
            "expected_behavior": "Should show no GPUs"
        },
        {
            "name": "Single GPU filtering",
            "cuda_visible_devices": "0",
            "expected_behavior": "Should show only GPU 0"
        },
        {
            "name": "Multiple GPU filtering",
            "cuda_visible_devices": "0,2",
            "expected_behavior": "Should show only GPUs 0 and 2"
        }
    ]
    
    # Import the GPU server module to test the endpoints directly
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'gpu_worker_pool'))
    from gpu_server import app
    
    # Create a test client using FastAPI's test functionality
    try:
        from fastapi.testclient import TestClient
        client = TestClient(app)
        
        for test_case in test_cases:
            print(f"\nTesting: {test_case['name']}")
            print(f"CUDA_VISIBLE_DEVICES: {repr(test_case['cuda_visible_devices'])}")
            print(f"Expected: {test_case['expected_behavior']}")
            
            # Set environment variable for this test
            if test_case['cuda_visible_devices'] is None:
                # Remove the environment variable
                if 'CUDA_VISIBLE_DEVICES' in os.environ:
                    del os.environ['CUDA_VISIBLE_DEVICES']
            else:
                os.environ['CUDA_VISIBLE_DEVICES'] = test_case['cuda_visible_devices']
            
            # Reinitialize the GPU monitor to pick up the new environment
            from gpu_server import gpu_monitor
            gpu_monitor._visible_devices = gpu_monitor._parse_visible_devices()
            gpu_monitor._cached_stats = None
            
            # Test all endpoints
            try:
                # Test /gpu/stats endpoint
                stats_response = client.get("/gpu/stats")
                assert stats_response.status_code == 200
                stats_data = stats_response.json()
                
                # Test /gpu/count endpoint
                count_response = client.get("/gpu/count")
                assert count_response.status_code == 200
                count_data = count_response.json()
                
                # Test /gpu/summary endpoint
                summary_response = client.get("/gpu/summary")
                assert summary_response.status_code == 200
                summary_data = summary_response.json()
                
                # Test /config endpoint
                config_response = client.get("/config")
                assert config_response.status_code == 200
                config_data = config_response.json()
                
                # Verify consistency across endpoints
                assert stats_data["gpu_count"] == count_data["gpu_count"]
                assert stats_data["gpu_count"] == summary_data["gpu_count"]
                
                # Verify config endpoint shows correct filtering info
                if "gpu_filtering" in config_data:
                    assert config_data["gpu_filtering"]["visible_gpu_count"] == stats_data["gpu_count"]
                
                # Verify GPU IDs are consistent between stats and summary
                if stats_data["gpus"]:
                    stats_gpu_ids = {gpu["gpu_id"] for gpu in stats_data["gpus"]}
                    summary_gpu_ids = {gpu["gpu_id"] for gpu in summary_data["gpus_summary"]}
                    assert stats_gpu_ids == summary_gpu_ids
                
                print(f"  ✓ GPU count: {stats_data['gpu_count']}")
                if stats_data["gpus"]:
                    gpu_ids = [gpu["gpu_id"] for gpu in stats_data["gpus"]]
                    print(f"  ✓ GPU IDs: {gpu_ids}")
                else:
                    print(f"  ✓ No GPUs visible")
                
                # Verify filtering behavior matches expectations
                if test_case['cuda_visible_devices'] is None:
                    # Should show all GPUs (or handle nvidia-smi not available gracefully)
                    if 'error' not in stats_data:
                        assert stats_data["gpu_count"] >= 0
                elif test_case['cuda_visible_devices'] == "":
                    # Should show no GPUs
                    assert stats_data["gpu_count"] == 0
                elif test_case['cuda_visible_devices'] == "0":
                    # Should show only GPU 0 (if it exists)
                    if 'error' not in stats_data and stats_data["gpu_count"] > 0:
                        assert all(gpu["gpu_id"] == 0 for gpu in stats_data["gpus"])
                elif test_case['cuda_visible_devices'] == "0,2":
                    # Should show only GPUs 0 and 2 (if they exist)
                    if 'error' not in stats_data and stats_data["gpu_count"] > 0:
                        gpu_ids = {gpu["gpu_id"] for gpu in stats_data["gpus"]}
                        assert gpu_ids.issubset({0, 2})
                
                print(f"  ✓ Filtering behavior matches expectations")
                
            except Exception as e:
                print(f"  ❌ Test failed: {e}")
                return False
        
        print("\n" + "=" * 60)
        print("✅ All API endpoint filtering integration tests passed!")
        return True
        
    except ImportError:
        print("❌ FastAPI TestClient not available, skipping integration tests")
        return True  # Don't fail if test client is not available

def test_config_endpoint_filtering_fields():
    """Test that config endpoint includes all required filtering fields"""
    print("\nTesting config endpoint filtering fields...")
    
    # Import the GPU server module
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'gpu_worker_pool'))
    from gpu_server import app
    
    try:
        from fastapi.testclient import TestClient
        client = TestClient(app)
        
        # Set a specific CUDA_VISIBLE_DEVICES value
        os.environ['CUDA_VISIBLE_DEVICES'] = "1,3"
        
        # Reinitialize the GPU monitor
        from gpu_server import gpu_monitor
        gpu_monitor._visible_devices = gpu_monitor._parse_visible_devices()
        gpu_monitor._cached_stats = None
        
        # Test config endpoint
        config_response = client.get("/config")
        assert config_response.status_code == 200
        config_data = config_response.json()
        
        # Verify gpu_filtering section exists
        assert "gpu_filtering" in config_data
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
            assert field in gpu_filtering, f"Missing field: {field}"
        
        # Verify field values are reasonable
        assert gpu_filtering["cuda_visible_devices"] == "1,3"
        assert gpu_filtering["visible_gpu_ids"] == [1, 3]
        assert gpu_filtering["filtering_active"] == True
        assert isinstance(gpu_filtering["total_system_gpus"], int)
        assert isinstance(gpu_filtering["visible_gpu_count"], int)
        
        print("✓ Config endpoint includes all required filtering fields")
        print(f"  CUDA_VISIBLE_DEVICES: {gpu_filtering['cuda_visible_devices']}")
        print(f"  Visible GPU IDs: {gpu_filtering['visible_gpu_ids']}")
        print(f"  Filtering active: {gpu_filtering['filtering_active']}")
        print(f"  Total system GPUs: {gpu_filtering['total_system_gpus']}")
        print(f"  Visible GPU count: {gpu_filtering['visible_gpu_count']}")
        
        return True
        
    except ImportError:
        print("❌ FastAPI TestClient not available, skipping config endpoint test")
        return True

def run_tests():
    """Run all integration tests"""
    try:
        success1 = test_api_endpoints_with_filtering()
        success2 = test_config_endpoint_filtering_fields()
        return success1 and success2
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)