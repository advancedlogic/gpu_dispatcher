#!/usr/bin/env python3
"""
Test script to verify the /config endpoint includes all required GPU filtering fields
"""

import sys
import os
sys.path.append('gpu_worker_pool')

try:
    from fastapi.testclient import TestClient
except ImportError:
    print("FastAPI TestClient not available, testing manually...")
    import requests
    import json
    
    def TestClient(app):
        class MockClient:
            def get(self, path):
                # For testing purposes, we'll simulate the response
                if path == '/config':
                    from gpu_server import get_config
                    import asyncio
                    result = asyncio.run(get_config())
                    class MockResponse:
                        def __init__(self, data):
                            self.data = data
                            self.status_code = 200
                        def json(self):
                            return self.data
                    return MockResponse(result)
        return MockClient()

from gpu_server import app

def test_config_endpoint():
    """Test that the /config endpoint includes all required GPU filtering fields"""
    client = TestClient(app)
    
    print("Testing /config endpoint...")
    
    # Test with no CUDA_VISIBLE_DEVICES set
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        del os.environ['CUDA_VISIBLE_DEVICES']
    
    response = client.get('/config')
    print(f"Status code: {response.status_code}")
    
    if response.status_code != 200:
        print(f"Error: {response.text}")
        return False
    
    config = response.json()
    gpu_filtering = config.get('gpu_filtering', {})
    
    print("GPU filtering section:")
    for key, value in gpu_filtering.items():
        print(f"  {key}: {value}")
    
    # Check all required fields are present
    required_fields = [
        'cuda_visible_devices',
        'visible_gpu_ids', 
        'filtering_active',
        'total_system_gpus',
        'visible_gpu_count'
    ]
    
    missing_fields = []
    for field in required_fields:
        if field not in gpu_filtering:
            missing_fields.append(field)
    
    if missing_fields:
        print(f"Missing required fields: {missing_fields}")
        return False
    
    print("✅ All required fields are present")
    
    # Test with CUDA_VISIBLE_DEVICES set
    print("\nTesting with CUDA_VISIBLE_DEVICES='0,1'...")
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    
    # Need to recreate the app to pick up the new environment variable
    from gpu_server import GPUMonitor
    monitor = GPUMonitor()
    
    # Check that the monitor parsed the environment variable correctly
    print(f"Parsed visible devices: {monitor._visible_devices}")
    print(f"Filtering active: {monitor._visible_devices is not None}")
    
    return True

if __name__ == "__main__":
    success = test_config_endpoint()
    if success:
        print("\n✅ Configuration endpoint test passed!")
    else:
        print("\n❌ Configuration endpoint test failed!")
        sys.exit(1)