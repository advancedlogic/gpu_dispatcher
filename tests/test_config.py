"""Unit tests for configuration management."""

import unittest
from unittest.mock import patch
from gpu_worker_pool.config import EnvironmentConfigurationManager


class TestEnvironmentConfigurationManager(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures."""
        # Clear any existing environment variables
        self.env_vars_to_clear = [
            'GPU_MEMORY_THRESHOLD_PERCENT',
            'GPU_UTILIZATION_THRESHOLD_PERCENT', 
            'GPU_POLLING_INTERVAL_SECONDS',
            'GPU_SERVICE_ENDPOINT'
        ]
        
    def tearDown(self):
        """Clean up after tests."""
        import os
        for var in self.env_vars_to_clear:
            if var in os.environ:
                del os.environ[var]
    
    def test_default_values_when_no_env_vars(self):
        """Test that default values are used when no environment variables are set."""
        config = EnvironmentConfigurationManager()
        
        self.assertEqual(config.get_memory_threshold(), 80.0)
        self.assertEqual(config.get_utilization_threshold(), 90.0)
        self.assertEqual(config.get_polling_interval(), 5)
        self.assertEqual(config.get_service_endpoint(), "http://localhost:8000")
    
    @patch.dict('os.environ', {
        'GPU_MEMORY_THRESHOLD_PERCENT': '75.5',
        'GPU_UTILIZATION_THRESHOLD_PERCENT': '85.0',
        'GPU_POLLING_INTERVAL_SECONDS': '10',
        'GPU_SERVICE_ENDPOINT': 'http://gpu-service:9090'
    })
    def test_valid_environment_variables(self):
        """Test loading valid values from environment variables."""
        config = EnvironmentConfigurationManager()
        
        self.assertEqual(config.get_memory_threshold(), 75.5)
        self.assertEqual(config.get_utilization_threshold(), 85.0)
        self.assertEqual(config.get_polling_interval(), 10)
        self.assertEqual(config.get_service_endpoint(), "http://gpu-service:9090")
    
    @patch.dict('os.environ', {'GPU_MEMORY_THRESHOLD_PERCENT': '150.0'})
    def test_invalid_threshold_too_high(self):
        """Test that invalid threshold values (too high) fall back to defaults."""
        config = EnvironmentConfigurationManager()
        self.assertEqual(config.get_memory_threshold(), 80.0)  # Should use default
    
    @patch.dict('os.environ', {'GPU_UTILIZATION_THRESHOLD_PERCENT': '-10.0'})
    def test_invalid_threshold_negative(self):
        """Test that invalid threshold values (negative) fall back to defaults."""
        config = EnvironmentConfigurationManager()
        self.assertEqual(config.get_utilization_threshold(), 90.0)  # Should use default
    
    @patch.dict('os.environ', {'GPU_MEMORY_THRESHOLD_PERCENT': 'not_a_number'})
    def test_invalid_threshold_format(self):
        """Test that invalid threshold format falls back to defaults."""
        config = EnvironmentConfigurationManager()
        self.assertEqual(config.get_memory_threshold(), 80.0)  # Should use default
    
    @patch.dict('os.environ', {'GPU_POLLING_INTERVAL_SECONDS': '0'})
    def test_invalid_polling_interval_zero(self):
        """Test that zero polling interval falls back to default."""
        config = EnvironmentConfigurationManager()
        self.assertEqual(config.get_polling_interval(), 5)  # Should use default
    
    @patch.dict('os.environ', {'GPU_POLLING_INTERVAL_SECONDS': '-5'})
    def test_invalid_polling_interval_negative(self):
        """Test that negative polling interval falls back to default."""
        config = EnvironmentConfigurationManager()
        self.assertEqual(config.get_polling_interval(), 5)  # Should use default
    
    @patch.dict('os.environ', {'GPU_POLLING_INTERVAL_SECONDS': 'invalid'})
    def test_invalid_polling_interval_format(self):
        """Test that invalid polling interval format falls back to default."""
        config = EnvironmentConfigurationManager()
        self.assertEqual(config.get_polling_interval(), 5)  # Should use default
    
    def test_boundary_threshold_values(self):
        """Test boundary values for thresholds (0 and 100)."""
        with patch.dict('os.environ', {'GPU_MEMORY_THRESHOLD_PERCENT': '0.0'}):
            config = EnvironmentConfigurationManager()
            self.assertEqual(config.get_memory_threshold(), 0.0)
        
        with patch.dict('os.environ', {'GPU_UTILIZATION_THRESHOLD_PERCENT': '100.0'}):
            config = EnvironmentConfigurationManager()
            self.assertEqual(config.get_utilization_threshold(), 100.0)
    
    @patch.dict('os.environ', {'GPU_POLLING_INTERVAL_SECONDS': '1'})
    def test_minimum_valid_polling_interval(self):
        """Test minimum valid polling interval."""
        config = EnvironmentConfigurationManager()
        self.assertEqual(config.get_polling_interval(), 1)


if __name__ == '__main__':
    unittest.main()