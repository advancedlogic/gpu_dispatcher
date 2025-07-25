#!/usr/bin/env python3
"""
Test script for CUDA_VISIBLE_DEVICES parsing functionality
"""

import os
import re
import sys
import logging
from typing import Optional, List
from unittest.mock import patch

# Configure logging to see the output
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_visible_devices() -> Optional[List[int]]:
    """
    Parse CUDA_VISIBLE_DEVICES environment variable to determine which GPUs should be visible.
    This is a standalone version of the method for testing.
    """
    cuda_visible = os.getenv("CUDA_VISIBLE_DEVICES")
    
    # Log the CUDA_VISIBLE_DEVICES value for debugging
    logger.info(f"CUDA_VISIBLE_DEVICES environment variable: {repr(cuda_visible)}")
    
    # If not set, show all GPUs (current behavior)
    if cuda_visible is None:
        logger.info("CUDA_VISIBLE_DEVICES not set, showing all GPUs")
        return None
    
    # If empty string, show no GPUs
    if cuda_visible.strip() == "":
        logger.info("CUDA_VISIBLE_DEVICES is empty, showing no GPUs")
        return []
    
    try:
        devices = []
        # Support comma-separated, space-separated, and mixed separators
        # Split on both commas and whitespace, then filter out empty strings
        parts = re.split(r'[,\s]+', cuda_visible.strip())
        
        for part in parts:
            part = part.strip()
            if not part:  # Skip empty parts
                continue
                
            try:
                device_id = int(part)
                if device_id < 0:
                    logger.warning(f"Negative GPU ID {device_id} in CUDA_VISIBLE_DEVICES ignored")
                    continue
                devices.append(device_id)
            except ValueError:
                logger.warning(f"Invalid GPU ID '{part}' in CUDA_VISIBLE_DEVICES ignored")
                continue
        
        if devices:
            logger.info(f"Parsed visible GPU IDs from CUDA_VISIBLE_DEVICES: {devices}")
            return devices
        else:
            logger.warning("No valid GPU IDs found in CUDA_VISIBLE_DEVICES, falling back to all GPUs")
            return None
            
    except Exception as e:
        logger.error(f"Error parsing CUDA_VISIBLE_DEVICES '{cuda_visible}': {e}")
        logger.info("Falling back to showing all GPUs")
        return None

def test_parse_visible_devices():
    """Test the _parse_visible_devices method with various inputs"""
    
    test_cases = [
        # (CUDA_VISIBLE_DEVICES value, expected result, description)
        (None, None, "Not set - should show all GPUs"),
        ("", [], "Empty string - should show no GPUs"),
        ("0", [0], "Single GPU - comma format"),
        ("0,1,2", [0, 1, 2], "Multiple GPUs - comma separated"),
        ("0 1 2", [0, 1, 2], "Multiple GPUs - space separated"),
        ("0,1 2", [0, 1, 2], "Multiple GPUs - mixed separators"),
        ("  0 , 1  ,  2  ", [0, 1, 2], "Multiple GPUs - with extra whitespace"),
        ("0,abc,2", [0, 2], "Mixed valid/invalid - should skip invalid"),
        ("abc", None, "All invalid - should fallback to all GPUs"),
        ("-1,0,1", [0, 1], "Negative ID - should skip negative"),
        ("0,,1", [0, 1], "Empty parts - should handle gracefully"),
        ("   ", [], "Only whitespace - should show no GPUs"),
    ]
    
    print("Testing CUDA_VISIBLE_DEVICES parsing functionality...")
    print("=" * 60)
    
    for cuda_visible, expected, description in test_cases:
        print(f"\nTest: {description}")
        print(f"Input: {repr(cuda_visible)}")
        print(f"Expected: {expected}")
        
        # Mock the environment variable
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': cuda_visible} if cuda_visible is not None else {}, clear=False):
            # Remove CUDA_VISIBLE_DEVICES if we're testing None case
            if cuda_visible is None and 'CUDA_VISIBLE_DEVICES' in os.environ:
                del os.environ['CUDA_VISIBLE_DEVICES']
            
            # Test the parsing function
            result = parse_visible_devices()
            
            print(f"Actual: {result}")
            
            if result == expected:
                print("✅ PASS")
            else:
                print("❌ FAIL")
                return False
    
    print("\n" + "=" * 60)
    print("All CUDA_VISIBLE_DEVICES parsing tests passed! ✅")
    return True

if __name__ == "__main__":
    success = test_parse_visible_devices()
    sys.exit(0 if success else 1)