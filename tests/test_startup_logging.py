#!/usr/bin/env python3
"""
Test script to demonstrate the enhanced startup logging for CUDA_VISIBLE_DEVICES filtering.
This shows the comprehensive startup information that is now logged.
"""

import os
import sys
import subprocess
from unittest.mock import patch

def test_startup_logging_scenarios():
    """Test different startup logging scenarios"""
    
    scenarios = [
        {
            "name": "No CUDA_VISIBLE_DEVICES set",
            "env": {},
            "description": "Should show all GPUs available"
        },
        {
            "name": "Empty CUDA_VISIBLE_DEVICES",
            "env": {"CUDA_VISIBLE_DEVICES": ""},
            "description": "Should show no GPUs"
        },
        {
            "name": "Valid GPU filtering",
            "env": {"CUDA_VISIBLE_DEVICES": "0,2"},
            "description": "Should show only GPUs 0 and 2"
        },
        {
            "name": "Mixed valid/invalid GPUs",
            "env": {"CUDA_VISIBLE_DEVICES": "0,abc,2,xyz"},
            "description": "Should show warnings and filter to valid GPUs"
        },
        {
            "name": "All invalid GPUs",
            "env": {"CUDA_VISIBLE_DEVICES": "abc,xyz,-1"},
            "description": "Should show errors and fallback to all GPUs"
        }
    ]
    
    print("Testing Enhanced Startup Logging for CUDA_VISIBLE_DEVICES Filtering")
    print("=" * 80)
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\nScenario {i}: {scenario['name']}")
        print(f"Description: {scenario['description']}")
        print(f"Environment: {scenario['env']}")
        print("-" * 60)
        
        # Create a test script that imports gpu_server to trigger startup logging
        test_script = f"""
import os
import sys
sys.path.insert(0, 'gpu_worker_pool')

# Set environment variables
{repr(scenario['env'])}
for key, value in {repr(scenario['env'])}.items():
    os.environ[key] = value

# Import to trigger startup logging
from gpu_server import GPUMonitor
monitor = GPUMonitor()

print("\\nMonitor created successfully!")
print(f"Visible devices: {{monitor._visible_devices}}")
"""
        
        # Write and execute the test script
        with open('temp_startup_test.py', 'w') as f:
            f.write(test_script)
        
        try:
            result = subprocess.run([sys.executable, 'temp_startup_test.py'], 
                                  capture_output=True, text=True, timeout=10)
            
            print("STDOUT:")
            print(result.stdout)
            if result.stderr:
                print("STDERR:")
                print(result.stderr)
                
        except subprocess.TimeoutExpired:
            print("Test timed out")
        except Exception as e:
            print(f"Error running test: {e}")
        
        # Clean up
        try:
            os.remove('temp_startup_test.py')
        except:
            pass
        
        print("-" * 60)

if __name__ == "__main__":
    test_startup_logging_scenarios()