#!/usr/bin/env python3
"""
Simple test script to verify the /config endpoint includes all required GPU filtering fields
"""

import sys
import os
import asyncio
sys.path.append('gpu_worker_pool')

def test_config_endpoint():
    """Test that the /config endpoint includes all required GPU filtering fields"""
    
    print("Testing /config endpoint...")
    
    # Import and call the config function directly
    from gpu_server import get_config
    
    # Test with no CUDA_VISIBLE_DEVICES set
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        del os.environ['CUDA_VISIBLE_DEVICES']
    
    # Call the config endpoint function
    config = asyncio.run(get_config())
    
    print("Configuration response received")
    
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
        print(f"❌ Missing required fields: {missing_fields}")
        return False
    
    print("✅ All required fields are present")
    
    # Verify field types and values make sense
    print("\nValidating field values:")
    
    # cuda_visible_devices should be None when not set
    if gpu_filtering['cuda_visible_devices'] is not None:
        print(f"⚠️  Expected cuda_visible_devices to be None, got: {gpu_filtering['cuda_visible_devices']}")
    else:
        print("✅ cuda_visible_devices is None (correct)")
    
    # visible_gpu_ids should be None when filtering is not active
    if gpu_filtering['visible_gpu_ids'] is not None:
        print(f"⚠️  Expected visible_gpu_ids to be None, got: {gpu_filtering['visible_gpu_ids']}")
    else:
        print("✅ visible_gpu_ids is None (correct)")
    
    # filtering_active should be False when no CUDA_VISIBLE_DEVICES is set
    if gpu_filtering['filtering_active'] != False:
        print(f"⚠️  Expected filtering_active to be False, got: {gpu_filtering['filtering_active']}")
    else:
        print("✅ filtering_active is False (correct)")
    
    # total_system_gpus should be a non-negative integer
    if not isinstance(gpu_filtering['total_system_gpus'], int) or gpu_filtering['total_system_gpus'] < 0:
        print(f"⚠️  Expected total_system_gpus to be non-negative int, got: {gpu_filtering['total_system_gpus']}")
    else:
        print(f"✅ total_system_gpus is valid: {gpu_filtering['total_system_gpus']}")
    
    # visible_gpu_count should be a non-negative integer
    if not isinstance(gpu_filtering['visible_gpu_count'], int) or gpu_filtering['visible_gpu_count'] < 0:
        print(f"⚠️  Expected visible_gpu_count to be non-negative int, got: {gpu_filtering['visible_gpu_count']}")
    else:
        print(f"✅ visible_gpu_count is valid: {gpu_filtering['visible_gpu_count']}")
    
    return True

if __name__ == "__main__":
    success = test_config_endpoint()
    if success:
        print("\n✅ Configuration endpoint test passed!")
    else:
        print("\n❌ Configuration endpoint test failed!")
        sys.exit(1)