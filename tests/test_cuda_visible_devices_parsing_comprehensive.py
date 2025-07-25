#!/usr/bin/env python3
"""
Comprehensive unit tests for CUDA_VISIBLE_DEVICES parsing functionality

This test file implements task 7 from the GPU visible devices filtering spec:
- Test parsing of comma-separated values ("0,1,2")
- Test parsing of space-separated values ("0 1 2")
- Test parsing of mixed separators ("0,1 2")
- Test handling of empty string and None values
- Test error handling for invalid values ("abc", "-1", mixed valid/invalid)
- Test whitespace handling and edge cases

Requirements tested: 5.1, 5.2, 5.3, 5.4, 5.5
"""

import os
import sys
import unittest
from unittest.mock import patch

# Add the gpu_worker_pool directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'gpu_worker_pool'))

from gpu_server import GPUMonitor


class TestCudaVisibleDevicesParsing(unittest.TestCase):
    """Comprehensive unit tests for CUDA_VISIBLE_DEVICES parsing"""
    
    def setUp(self):
        """Set up test environment"""
        # Clear any existing CUDA_VISIBLE_DEVICES to ensure clean state
        if 'CUDA_VISIBLE_DEVICES' in os.environ:
            del os.environ['CUDA_VISIBLE_DEVICES']
    
    def tearDown(self):
        """Clean up test environment"""
        # Clear CUDA_VISIBLE_DEVICES after each test
        if 'CUDA_VISIBLE_DEVICES' in os.environ:
            del os.environ['CUDA_VISIBLE_DEVICES']
    
    # Test parsing of comma-separated values ("0,1,2") - Requirement 5.1
    def test_comma_separated_single_gpu(self):
        """Test parsing single GPU with comma format"""
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '0'}):
            monitor = GPUMonitor()
            result = monitor._parse_visible_devices()
            self.assertEqual(result, [0])
    
    def test_comma_separated_multiple_gpus(self):
        """Test parsing multiple GPUs with comma-separated format"""
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '0,1,2'}):
            monitor = GPUMonitor()
            result = monitor._parse_visible_devices()
            self.assertEqual(result, [0, 1, 2])
    
    def test_comma_separated_non_sequential(self):
        """Test parsing non-sequential GPU IDs with commas"""
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '0,2,5'}):
            monitor = GPUMonitor()
            result = monitor._parse_visible_devices()
            self.assertEqual(result, [0, 2, 5])
    
    def test_comma_separated_single_digit_high_numbers(self):
        """Test parsing high GPU numbers with commas"""
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '7,8,9'}):
            monitor = GPUMonitor()
            result = monitor._parse_visible_devices()
            self.assertEqual(result, [7, 8, 9])
    
    # Test parsing of space-separated values ("0 1 2") - Requirement 5.2
    def test_space_separated_single_gpu(self):
        """Test parsing single GPU with space format"""
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '1'}):
            monitor = GPUMonitor()
            result = monitor._parse_visible_devices()
            self.assertEqual(result, [1])
    
    def test_space_separated_multiple_gpus(self):
        """Test parsing multiple GPUs with space-separated format"""
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '0 1 2'}):
            monitor = GPUMonitor()
            result = monitor._parse_visible_devices()
            self.assertEqual(result, [0, 1, 2])
    
    def test_space_separated_non_sequential(self):
        """Test parsing non-sequential GPU IDs with spaces"""
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '1 3 7'}):
            monitor = GPUMonitor()
            result = monitor._parse_visible_devices()
            self.assertEqual(result, [1, 3, 7])
    
    def test_space_separated_multiple_spaces(self):
        """Test parsing with multiple spaces between GPU IDs"""
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '0  1   2'}):
            monitor = GPUMonitor()
            result = monitor._parse_visible_devices()
            self.assertEqual(result, [0, 1, 2])
    
    # Test parsing of mixed separators ("0,1 2") - Requirement 5.3
    def test_mixed_separators_comma_space(self):
        """Test parsing with mixed comma and space separators"""
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '0,1 2'}):
            monitor = GPUMonitor()
            result = monitor._parse_visible_devices()
            self.assertEqual(result, [0, 1, 2])
    
    def test_mixed_separators_space_comma(self):
        """Test parsing with space then comma separators"""
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '0 1,2'}):
            monitor = GPUMonitor()
            result = monitor._parse_visible_devices()
            self.assertEqual(result, [0, 1, 2])
    
    def test_mixed_separators_complex(self):
        """Test parsing with complex mixed separators"""
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '0,1 2,3 4'}):
            monitor = GPUMonitor()
            result = monitor._parse_visible_devices()
            self.assertEqual(result, [0, 1, 2, 3, 4])
    
    def test_mixed_separators_with_extra_spaces(self):
        """Test parsing mixed separators with extra whitespace"""
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '0 , 1  2 , 3'}):
            monitor = GPUMonitor()
            result = monitor._parse_visible_devices()
            self.assertEqual(result, [0, 1, 2, 3])
    
    # Test handling of empty string and None values - Requirement 5.4
    def test_empty_string(self):
        """Test parsing empty string returns empty list"""
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': ''}):
            monitor = GPUMonitor()
            result = monitor._parse_visible_devices()
            self.assertEqual(result, [])
    
    def test_whitespace_only_string(self):
        """Test parsing whitespace-only string returns empty list"""
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '   '}):
            monitor = GPUMonitor()
            result = monitor._parse_visible_devices()
            self.assertEqual(result, [])
    
    def test_tabs_and_spaces_only(self):
        """Test parsing tabs and spaces only returns empty list"""
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': ' \t \n '}):
            monitor = GPUMonitor()
            result = monitor._parse_visible_devices()
            self.assertEqual(result, [])
    
    def test_none_value_not_set(self):
        """Test parsing when CUDA_VISIBLE_DEVICES is not set returns None"""
        # Ensure environment variable is not set
        if 'CUDA_VISIBLE_DEVICES' in os.environ:
            del os.environ['CUDA_VISIBLE_DEVICES']
        
        monitor = GPUMonitor()
        result = monitor._parse_visible_devices()
        self.assertIsNone(result)
    
    # Test error handling for invalid values - Requirement 5.5
    def test_invalid_alphabetic_values(self):
        """Test parsing with alphabetic characters falls back to None"""
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': 'abc'}):
            monitor = GPUMonitor()
            result = monitor._parse_visible_devices()
            self.assertIsNone(result)
    
    def test_invalid_mixed_alphabetic_numeric(self):
        """Test parsing with mixed alphabetic and numeric characters"""
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': 'a1b2c'}):
            monitor = GPUMonitor()
            result = monitor._parse_visible_devices()
            self.assertIsNone(result)
    
    def test_negative_values_filtered_out(self):
        """Test parsing with negative values filters them out"""
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '-1'}):
            monitor = GPUMonitor()
            result = monitor._parse_visible_devices()
            self.assertIsNone(result)  # All invalid, falls back to None
    
    def test_negative_mixed_with_valid(self):
        """Test parsing negative values mixed with valid ones"""
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '-1,0,1'}):
            monitor = GPUMonitor()
            result = monitor._parse_visible_devices()
            self.assertEqual(result, [0, 1])  # Negative filtered out, valid ones kept
    
    def test_mixed_valid_invalid_values(self):
        """Test parsing with mix of valid and invalid values"""
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '0,abc,2'}):
            monitor = GPUMonitor()
            result = monitor._parse_visible_devices()
            self.assertEqual(result, [0, 2])  # Invalid 'abc' filtered out
    
    def test_mixed_valid_invalid_negative(self):
        """Test parsing with valid, invalid, and negative values"""
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '0,abc,-1,2,xyz'}):
            monitor = GPUMonitor()
            result = monitor._parse_visible_devices()
            self.assertEqual(result, [0, 2])  # Only valid positive integers kept
    
    def test_special_characters(self):
        """Test parsing with special characters"""
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '0,@,#,$,1'}):
            monitor = GPUMonitor()
            result = monitor._parse_visible_devices()
            self.assertEqual(result, [0, 1])  # Special characters filtered out
    
    def test_floating_point_values(self):
        """Test parsing with floating point values"""
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '0.5,1.2,2'}):
            monitor = GPUMonitor()
            result = monitor._parse_visible_devices()
            self.assertEqual(result, [2])  # Only integer value kept
    
    # Test whitespace handling and edge cases - Additional comprehensive tests
    def test_leading_trailing_whitespace(self):
        """Test parsing with leading and trailing whitespace"""
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '  0,1,2  '}):
            monitor = GPUMonitor()
            result = monitor._parse_visible_devices()
            self.assertEqual(result, [0, 1, 2])
    
    def test_whitespace_around_separators(self):
        """Test parsing with whitespace around separators"""
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '0 , 1 , 2'}):
            monitor = GPUMonitor()
            result = monitor._parse_visible_devices()
            self.assertEqual(result, [0, 1, 2])
    
    def test_empty_parts_between_separators(self):
        """Test parsing with empty parts between separators"""
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '0,,1,2'}):
            monitor = GPUMonitor()
            result = monitor._parse_visible_devices()
            self.assertEqual(result, [0, 1, 2])
    
    def test_multiple_empty_parts(self):
        """Test parsing with multiple empty parts"""
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '0,,,1,,2,'}):
            monitor = GPUMonitor()
            result = monitor._parse_visible_devices()
            self.assertEqual(result, [0, 1, 2])
    
    def test_mixed_empty_parts_and_spaces(self):
        """Test parsing with mixed empty parts and spaces"""
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '0, , 1  ,, 2'}):
            monitor = GPUMonitor()
            result = monitor._parse_visible_devices()
            self.assertEqual(result, [0, 1, 2])
    
    def test_duplicate_gpu_ids(self):
        """Test parsing with duplicate GPU IDs"""
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '0,1,0,2,1'}):
            monitor = GPUMonitor()
            result = monitor._parse_visible_devices()
            self.assertEqual(result, [0, 1, 0, 2, 1])  # Duplicates preserved as per CUDA behavior
    
    def test_large_gpu_numbers(self):
        """Test parsing with large GPU numbers"""
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '15,31,63'}):
            monitor = GPUMonitor()
            result = monitor._parse_visible_devices()
            self.assertEqual(result, [15, 31, 63])
    
    def test_single_character_invalid(self):
        """Test parsing with single invalid character"""
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': 'x'}):
            monitor = GPUMonitor()
            result = monitor._parse_visible_devices()
            self.assertIsNone(result)
    
    def test_only_separators(self):
        """Test parsing with only separators falls back to None"""
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': ',, ,,'}):
            monitor = GPUMonitor()
            result = monitor._parse_visible_devices()
            self.assertIsNone(result)  # No valid GPUs found, falls back to None
    
    def test_newlines_and_tabs(self):
        """Test parsing with newlines and tabs"""
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '0\n1\t2'}):
            monitor = GPUMonitor()
            result = monitor._parse_visible_devices()
            self.assertEqual(result, [0, 1, 2])


def run_comprehensive_parsing_tests():
    """Run all comprehensive CUDA_VISIBLE_DEVICES parsing tests"""
    print("Running comprehensive CUDA_VISIBLE_DEVICES parsing tests...")
    print("=" * 80)
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestCudaVisibleDevicesParsing)
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)
    
    print("=" * 80)
    
    if result.wasSuccessful():
        print(f"✅ All {result.testsRun} CUDA_VISIBLE_DEVICES parsing tests passed!")
        return True
    else:
        print(f"❌ {len(result.failures + result.errors)} out of {result.testsRun} tests failed!")
        return False


if __name__ == "__main__":
    success = run_comprehensive_parsing_tests()
    sys.exit(0 if success else 1)