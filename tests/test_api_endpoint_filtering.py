#!/usr/bin/env python3
"""
Test the /config API endpoint via HTTP to ensure it returns the GPU filtering information
"""

import os
import sys
import subprocess
import time
import requests
import json

def start_server_and_test():
    """Start the GPU server and test the /config endpoint via HTTP"""
    
    print("Starting GPU server for API testing...")
    
    # Set environment variable for testing
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = '0,1'
    env['PORT'] = '8001'  # Use different port to avoid conflicts
    env['LOG_LEVEL'] = 'WARNING'  # Reduce log noise
    
    # Start the server
    server_process = subprocess.Popen(
        [sys.executable, 'gpu_worker_pool/gpu_server.py'],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    try:
        # Wait for server to start
        print("Waiting for server to start...")
        time.sleep(3)
        
        # Test the /config endpoint
        print("Testing /config endpoint via HTTP...")
        
        try:
            response = requests.get('http://localhost:8001/config', timeout=10)
            
            if response.status_code != 200:
                print(f"❌ HTTP request failed with status {response.status_code}")
                print(f"Response: {response.text}")
                return False
            
            config = response.json()
            
            # Check if gpu_filtering section exists
            if 'gpu_filtering' not in config:
                print("❌ gpu_filtering section missing from config response")
                return False
            
            gpu_filtering = config['gpu_filtering']
            
            print("✅ HTTP request successful")
            print("GPU filtering section from HTTP response:")
            for key, value in gpu_filtering.items():
                print(f"  {key}: {value}")
            
            # Verify all required fields
            required_fields = [
                'cuda_visible_devices',
                'visible_gpu_ids', 
                'filtering_active',
                'total_system_gpus',
                'visible_gpu_count'
            ]
            
            missing_fields = [field for field in required_fields if field not in gpu_filtering]
            
            if missing_fields:
                print(f"❌ Missing required fields: {missing_fields}")
                return False
            
            print("✅ All required fields present in HTTP response")
            
            # Verify values make sense
            if gpu_filtering['cuda_visible_devices'] != '0,1':
                print(f"❌ Expected cuda_visible_devices='0,1', got: {gpu_filtering['cuda_visible_devices']}")
                return False
            
            if gpu_filtering['visible_gpu_ids'] != [0, 1]:
                print(f"❌ Expected visible_gpu_ids=[0, 1], got: {gpu_filtering['visible_gpu_ids']}")
                return False
            
            if gpu_filtering['filtering_active'] != True:
                print(f"❌ Expected filtering_active=True, got: {gpu_filtering['filtering_active']}")
                return False
            
            print("✅ All field values are correct")
            return True
            
        except requests.exceptions.RequestException as e:
            print(f"❌ HTTP request failed: {e}")
            return False
            
    finally:
        # Clean up server process
        print("Stopping server...")
        server_process.terminate()
        try:
            server_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server_process.kill()
            server_process.wait()

def main():
    """Main test function"""
    print("Testing /config endpoint via HTTP API...")
    
    success = start_server_and_test()
    
    if success:
        print("\n🎉 HTTP API test passed!")
        return True
    else:
        print("\n❌ HTTP API test failed!")
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)