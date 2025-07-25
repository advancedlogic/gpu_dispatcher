#!/usr/bin/env python3
"""
Test script to verify the /config endpoint includes all required GPU filtering fields
with different CUDA_VISIBLE_DEVICES configurations
"""

import os
import sys
import subprocess

def test_config_with_env(cuda_visible_devices_value):
    """Test config endpoint with a specific CUDA_VISIBLE_DEVICES value"""
    
    print(f"\n=== Testing with CUDA_VISIBLE_DEVICES='{cuda_visible_devices_value}' ===")
    
    # Create a test script that sets the environment variable before importing
    test_script = f'''
import os
import sys
import asyncio

# Set environment variable before importing
if "{cuda_visible_devices_value}" == "None":
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        del os.environ["CUDA_VISIBLE_DEVICES"]
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = "{cuda_visible_devices_value}"

sys.path.append("gpu_worker_pool")

from gpu_server import get_config

# Call the config endpoint function
config = asyncio.run(get_config())

gpu_filtering = config.get("gpu_filtering", {{}})

print("GPU filtering section:")
for key, value in gpu_filtering.items():
    print(f"  {{key}}: {{value}}")

# Check all required fields are present
required_fields = [
    "cuda_visible_devices",
    "visible_gpu_ids", 
    "filtering_active",
    "total_system_gpus",
    "visible_gpu_count"
]

missing_fields = []
for field in required_fields:
    if field not in gpu_filtering:
        missing_fields.append(field)

if missing_fields:
    print(f"❌ Missing required fields: {{missing_fields}}")
    exit(1)

print("✅ All required fields are present")

# Verify the cuda_visible_devices field matches what we set
expected_cuda_visible = None if "{cuda_visible_devices_value}" == "None" else "{cuda_visible_devices_value}"
actual_cuda_visible = gpu_filtering["cuda_visible_devices"]

if actual_cuda_visible == expected_cuda_visible:
    print(f"✅ cuda_visible_devices matches expected value: {{actual_cuda_visible}}")
else:
    print(f"❌ cuda_visible_devices mismatch. Expected: {{expected_cuda_visible}}, Got: {{actual_cuda_visible}}")
    exit(1)

print("✅ Test passed!")
'''
    
    # Write and run the test script
    with open('temp_test.py', 'w') as f:
        f.write(test_script)
    
    try:
        result = subprocess.run([sys.executable, 'temp_test.py'], 
                              capture_output=True, text=True, timeout=30)
        
        print("STDOUT:")
        print(result.stdout)
        
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        
        if result.returncode != 0:
            print(f"❌ Test failed with return code: {result.returncode}")
            return False
        
        return True
        
    except subprocess.TimeoutExpired:
        print("❌ Test timed out")
        return False
    finally:
        # Clean up temp file
        if os.path.exists('temp_test.py'):
            os.remove('temp_test.py')

def main():
    """Run tests with different CUDA_VISIBLE_DEVICES configurations"""
    
    print("Testing /config endpoint GPU filtering fields...")
    
    test_cases = [
        "None",  # No environment variable set
        "",      # Empty string
        "0",     # Single GPU
        "0,1",   # Multiple GPUs comma-separated
        "0 1",   # Multiple GPUs space-separated
        "0,1 2", # Mixed separators
    ]
    
    all_passed = True
    
    for test_case in test_cases:
        if not test_config_with_env(test_case):
            all_passed = False
    
    if all_passed:
        print("\n🎉 All configuration endpoint tests passed!")
        return True
    else:
        print("\n❌ Some configuration endpoint tests failed!")
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)