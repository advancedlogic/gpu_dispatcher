"""Configuration management for GPU Worker Pool."""

import os
import logging
from abc import ABC, abstractmethod
from typing import Optional, List
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


class ConfigurationManager(ABC):
    """Abstract base class for configuration management."""
    
    @abstractmethod
    def get_memory_threshold(self) -> float:
        """Get the GPU memory usage threshold percentage."""
        pass
    
    @abstractmethod
    def get_utilization_threshold(self) -> float:
        """Get the GPU utilization threshold percentage."""
        pass
    
    @abstractmethod
    def get_polling_interval(self) -> int:
        """Get the polling interval in seconds."""
        pass
    
    @abstractmethod
    def get_service_endpoint(self) -> str:
        """Get the GPU statistics service endpoint."""
        pass


class MultiEndpointConfigurationManager(ABC):
    """Abstract base class for multi-endpoint configuration management."""
    
    @abstractmethod
    def get_memory_threshold(self) -> float:
        """Get the GPU memory usage threshold percentage."""
        pass
    
    @abstractmethod
    def get_utilization_threshold(self) -> float:
        """Get the GPU utilization threshold percentage."""
        pass
    
    @abstractmethod
    def get_polling_interval(self) -> int:
        """Get the polling interval in seconds."""
        pass
    
    @abstractmethod
    def get_service_endpoints(self) -> List[str]:
        """Get the list of GPU statistics service endpoints."""
        pass
    
    @abstractmethod
    def get_load_balancing_strategy(self) -> str:
        """Get the load balancing strategy."""
        pass
    
    @abstractmethod
    def get_endpoint_timeout(self) -> float:
        """Get the endpoint timeout in seconds."""
        pass
    
    @abstractmethod
    def get_endpoint_max_retries(self) -> int:
        """Get the maximum number of retries per endpoint."""
        pass
    
    @abstractmethod
    def get_endpoint_health_check_interval(self) -> int:
        """Get the endpoint health check interval in seconds."""
        pass


class EnvironmentConfigurationManager(ConfigurationManager):
    """Configuration manager that reads from environment variables."""
    
    # Default values as specified in requirements
    DEFAULT_MEMORY_THRESHOLD = 80.0
    DEFAULT_UTILIZATION_THRESHOLD = 90.0
    DEFAULT_POLLING_INTERVAL = 5
    DEFAULT_SERVICE_ENDPOINT = "http://localhost:8080"
    
    def __init__(self):
        """Initialize configuration manager and load values."""
        self._memory_threshold = self._load_threshold(
            "GPU_MEMORY_THRESHOLD_PERCENT", 
            self.DEFAULT_MEMORY_THRESHOLD
        )
        self._utilization_threshold = self._load_threshold(
            "GPU_UTILIZATION_THRESHOLD_PERCENT", 
            self.DEFAULT_UTILIZATION_THRESHOLD
        )
        self._polling_interval = self._load_polling_interval()
        self._service_endpoint = self._load_service_endpoint()
    
    def get_memory_threshold(self) -> float:
        """Get the GPU memory usage threshold percentage."""
        return self._memory_threshold
    
    def get_utilization_threshold(self) -> float:
        """Get the GPU utilization threshold percentage."""
        return self._utilization_threshold
    
    def get_polling_interval(self) -> int:
        """Get the polling interval in seconds."""
        return self._polling_interval
    
    def get_service_endpoint(self) -> str:
        """Get the GPU statistics service endpoint."""
        return self._service_endpoint
    
    def _load_threshold(self, env_var: str, default_value: float) -> float:
        """Load and validate a threshold value from environment variable."""
        env_value = os.getenv(env_var)
        
        if env_value is None:
            logger.info(f"{env_var} not set, using default value: {default_value}%")
            return default_value
        
        try:
            threshold = float(env_value)
            if not self._is_valid_threshold(threshold):
                logger.error(
                    f"Invalid {env_var} value: {threshold}. Must be between 0 and 100. "
                    f"Using default value: {default_value}%"
                )
                return default_value
            
            logger.info(f"Loaded {env_var}: {threshold}%")
            return threshold
            
        except ValueError:
            logger.error(
                f"Invalid {env_var} format: '{env_value}'. Must be a number. "
                f"Using default value: {default_value}%"
            )
            return default_value
    
    def _load_polling_interval(self) -> int:
        """Load polling interval from environment variable."""
        env_value = os.getenv("GPU_POLLING_INTERVAL_SECONDS")
        
        if env_value is None:
            return self.DEFAULT_POLLING_INTERVAL
        
        try:
            interval = int(env_value)
            if interval <= 0:
                logger.error(
                    f"Invalid GPU_POLLING_INTERVAL_SECONDS: {interval}. Must be positive. "
                    f"Using default: {self.DEFAULT_POLLING_INTERVAL}"
                )
                return self.DEFAULT_POLLING_INTERVAL
            
            return interval
            
        except ValueError:
            logger.error(
                f"Invalid GPU_POLLING_INTERVAL_SECONDS format: '{env_value}'. "
                f"Using default: {self.DEFAULT_POLLING_INTERVAL}"
            )
            return self.DEFAULT_POLLING_INTERVAL
    
    def _load_service_endpoint(self) -> str:
        """Load service endpoint from environment variable."""
        endpoint = os.getenv("GPU_SERVICE_ENDPOINT", self.DEFAULT_SERVICE_ENDPOINT)
        return endpoint
    
    @staticmethod
    def _is_valid_threshold(value: float) -> bool:
        """Validate that threshold value is between 0 and 100."""
        return 0.0 <= value <= 100.0


class EnvironmentMultiEndpointConfigurationManager(MultiEndpointConfigurationManager):
    """Multi-endpoint configuration manager that reads from environment variables."""
    
    # Default values for multi-endpoint configuration
    DEFAULT_MEMORY_THRESHOLD = 80.0
    DEFAULT_UTILIZATION_THRESHOLD = 90.0
    DEFAULT_POLLING_INTERVAL = 5
    DEFAULT_SERVICE_ENDPOINT = "http://localhost:8000"  # Updated default port
    DEFAULT_LOAD_BALANCING_STRATEGY = "availability"
    DEFAULT_ENDPOINT_TIMEOUT = 10.0
    DEFAULT_ENDPOINT_MAX_RETRIES = 3
    DEFAULT_ENDPOINT_HEALTH_CHECK_INTERVAL = 30
    
    # Valid load balancing strategies
    VALID_LOAD_BALANCING_STRATEGIES = {"availability", "round_robin", "weighted"}
    
    def __init__(self, service_endpoints: Optional[List[str]] = None):
        """Initialize multi-endpoint configuration manager and load values.
        
        Args:
            service_endpoints: Optional list of service endpoints to override environment config
        """
        self._memory_threshold = self._load_threshold(
            "GPU_MEMORY_THRESHOLD_PERCENT", 
            self.DEFAULT_MEMORY_THRESHOLD
        )
        self._utilization_threshold = self._load_threshold(
            "GPU_UTILIZATION_THRESHOLD_PERCENT", 
            self.DEFAULT_UTILIZATION_THRESHOLD
        )
        self._polling_interval = self._load_polling_interval()
        self._service_endpoints = self._load_service_endpoints(service_endpoints)
        self._load_balancing_strategy = self._load_load_balancing_strategy()
        self._endpoint_timeout = self._load_endpoint_timeout()
        self._endpoint_max_retries = self._load_endpoint_max_retries()
        self._endpoint_health_check_interval = self._load_endpoint_health_check_interval()
    
    def get_memory_threshold(self) -> float:
        """Get the GPU memory usage threshold percentage."""
        return self._memory_threshold
    
    def get_utilization_threshold(self) -> float:
        """Get the GPU utilization threshold percentage."""
        return self._utilization_threshold
    
    def get_polling_interval(self) -> int:
        """Get the polling interval in seconds."""
        return self._polling_interval
    
    def get_service_endpoints(self) -> List[str]:
        """Get the list of GPU statistics service endpoints."""
        return self._service_endpoints.copy()  # Return a copy to prevent modification
    
    def get_load_balancing_strategy(self) -> str:
        """Get the load balancing strategy."""
        return self._load_balancing_strategy
    
    def get_endpoint_timeout(self) -> float:
        """Get the endpoint timeout in seconds."""
        return self._endpoint_timeout
    
    def get_endpoint_max_retries(self) -> int:
        """Get the maximum number of retries per endpoint."""
        return self._endpoint_max_retries
    
    def get_endpoint_health_check_interval(self) -> int:
        """Get the endpoint health check interval in seconds."""
        return self._endpoint_health_check_interval
    
    def _load_threshold(self, env_var: str, default_value: float) -> float:
        """Load and validate a threshold value from environment variable."""
        env_value = os.getenv(env_var)
        
        if env_value is None:
            logger.info(f"{env_var} not set, using default value: {default_value}%")
            return default_value
        
        try:
            threshold = float(env_value)
            if not self._is_valid_threshold(threshold):
                logger.error(
                    f"Invalid {env_var} value: {threshold}. Must be between 0 and 100. "
                    f"Using default value: {default_value}%"
                )
                return default_value
            
            logger.info(f"Loaded {env_var}: {threshold}%")
            return threshold
            
        except ValueError:
            logger.error(
                f"Invalid {env_var} format: '{env_value}'. Must be a number. "
                f"Using default value: {default_value}%"
            )
            return default_value
    
    def _load_polling_interval(self) -> int:
        """Load polling interval from environment variable."""
        env_value = os.getenv("GPU_POLLING_INTERVAL_SECONDS")
        
        if env_value is None:
            return self.DEFAULT_POLLING_INTERVAL
        
        try:
            interval = int(env_value)
            if interval <= 0:
                logger.error(
                    f"Invalid GPU_POLLING_INTERVAL_SECONDS: {interval}. Must be positive. "
                    f"Using default: {self.DEFAULT_POLLING_INTERVAL}"
                )
                return self.DEFAULT_POLLING_INTERVAL
            
            return interval
            
        except ValueError:
            logger.error(
                f"Invalid GPU_POLLING_INTERVAL_SECONDS format: '{env_value}'. "
                f"Using default: {self.DEFAULT_POLLING_INTERVAL}"
            )
            return self.DEFAULT_POLLING_INTERVAL
    
    def _load_service_endpoints(self, override_endpoints: Optional[List[str]] = None) -> List[str]:
        """Load service endpoints from environment variables or override.
        
        Args:
            override_endpoints: Optional list of endpoints to use instead of environment config
            
        Returns:
            List of validated service endpoints
        """
        # Use override endpoints if provided
        if override_endpoints is not None:
            if not isinstance(override_endpoints, list):
                logger.error(f"override_endpoints must be a list, got {type(override_endpoints)}")
                return [self.DEFAULT_SERVICE_ENDPOINT]
            
            if not override_endpoints:
                logger.warning("Empty override_endpoints list provided, using default endpoint")
                return [self.DEFAULT_SERVICE_ENDPOINT]
            
            return self._validate_and_deduplicate_endpoints(override_endpoints)
        
        # Check for multiple endpoints first (new format)
        multi_endpoints_env = os.getenv("GPU_SERVICE_ENDPOINTS")
        if multi_endpoints_env:
            endpoints = [endpoint.strip() for endpoint in multi_endpoints_env.split(',')]
            endpoints = [endpoint for endpoint in endpoints if endpoint]  # Remove empty strings
            
            if not endpoints:
                logger.warning("GPU_SERVICE_ENDPOINTS is empty, falling back to single endpoint")
            else:
                logger.info(f"Loaded {len(endpoints)} endpoints from GPU_SERVICE_ENDPOINTS")
                return self._validate_and_deduplicate_endpoints(endpoints)
        
        # Fall back to single endpoint (backward compatibility)
        single_endpoint = os.getenv("GPU_SERVICE_ENDPOINT", self.DEFAULT_SERVICE_ENDPOINT)
        logger.info(f"Using single endpoint from GPU_SERVICE_ENDPOINT: {single_endpoint}")
        return self._validate_and_deduplicate_endpoints([single_endpoint])
    
    def _validate_and_deduplicate_endpoints(self, endpoints: List[str]) -> List[str]:
        """Validate and deduplicate a list of endpoints.
        
        Args:
            endpoints: List of endpoint URLs to validate
            
        Returns:
            List of validated and deduplicated endpoints
        """
        validated_endpoints = []
        seen_endpoints = set()
        
        for endpoint in endpoints:
            if not isinstance(endpoint, str):
                logger.warning(f"Skipping non-string endpoint: {endpoint}")
                continue
            
            endpoint = endpoint.strip()
            if not endpoint:
                logger.warning("Skipping empty endpoint")
                continue
            
            # Validate URL format
            try:
                parsed = urlparse(endpoint)
                if not parsed.scheme or not parsed.netloc:
                    logger.warning(f"Skipping invalid endpoint URL: {endpoint}")
                    continue
                
                # Normalize the endpoint URL
                normalized_endpoint = f"{parsed.scheme}://{parsed.netloc}{parsed.path}".rstrip('/')
                
                # Check for duplicates
                if normalized_endpoint in seen_endpoints:
                    logger.info(f"Skipping duplicate endpoint: {endpoint}")
                    continue
                
                seen_endpoints.add(normalized_endpoint)
                validated_endpoints.append(normalized_endpoint)
                logger.debug(f"Validated endpoint: {normalized_endpoint}")
                
            except Exception as e:
                logger.warning(f"Skipping invalid endpoint {endpoint}: {e}")
                continue
        
        if not validated_endpoints:
            logger.error("No valid endpoints found, using default endpoint")
            return [self.DEFAULT_SERVICE_ENDPOINT]
        
        logger.info(f"Loaded {len(validated_endpoints)} valid endpoints")
        return validated_endpoints
    
    def _load_load_balancing_strategy(self) -> str:
        """Load load balancing strategy from environment variable."""
        strategy = os.getenv("GPU_LOAD_BALANCING_STRATEGY", self.DEFAULT_LOAD_BALANCING_STRATEGY).lower()
        
        if strategy not in self.VALID_LOAD_BALANCING_STRATEGIES:
            logger.warning(
                f"Invalid GPU_LOAD_BALANCING_STRATEGY: {strategy}. "
                f"Valid options: {', '.join(self.VALID_LOAD_BALANCING_STRATEGIES)}. "
                f"Using default: {self.DEFAULT_LOAD_BALANCING_STRATEGY}"
            )
            return self.DEFAULT_LOAD_BALANCING_STRATEGY
        
        logger.info(f"Loaded load balancing strategy: {strategy}")
        return strategy
    
    def _load_endpoint_timeout(self) -> float:
        """Load endpoint timeout from environment variable."""
        env_value = os.getenv("GPU_ENDPOINT_TIMEOUT")
        
        if env_value is None:
            return self.DEFAULT_ENDPOINT_TIMEOUT
        
        try:
            timeout = float(env_value)
            if timeout <= 0:
                logger.error(
                    f"Invalid GPU_ENDPOINT_TIMEOUT: {timeout}. Must be positive. "
                    f"Using default: {self.DEFAULT_ENDPOINT_TIMEOUT}"
                )
                return self.DEFAULT_ENDPOINT_TIMEOUT
            
            logger.info(f"Loaded endpoint timeout: {timeout}s")
            return timeout
            
        except ValueError:
            logger.error(
                f"Invalid GPU_ENDPOINT_TIMEOUT format: '{env_value}'. "
                f"Using default: {self.DEFAULT_ENDPOINT_TIMEOUT}"
            )
            return self.DEFAULT_ENDPOINT_TIMEOUT
    
    def _load_endpoint_max_retries(self) -> int:
        """Load endpoint max retries from environment variable."""
        env_value = os.getenv("GPU_ENDPOINT_MAX_RETRIES")
        
        if env_value is None:
            return self.DEFAULT_ENDPOINT_MAX_RETRIES
        
        try:
            max_retries = int(env_value)
            if max_retries < 0:
                logger.error(
                    f"Invalid GPU_ENDPOINT_MAX_RETRIES: {max_retries}. Must be non-negative. "
                    f"Using default: {self.DEFAULT_ENDPOINT_MAX_RETRIES}"
                )
                return self.DEFAULT_ENDPOINT_MAX_RETRIES
            
            logger.info(f"Loaded endpoint max retries: {max_retries}")
            return max_retries
            
        except ValueError:
            logger.error(
                f"Invalid GPU_ENDPOINT_MAX_RETRIES format: '{env_value}'. "
                f"Using default: {self.DEFAULT_ENDPOINT_MAX_RETRIES}"
            )
            return self.DEFAULT_ENDPOINT_MAX_RETRIES
    
    def _load_endpoint_health_check_interval(self) -> int:
        """Load endpoint health check interval from environment variable."""
        env_value = os.getenv("GPU_ENDPOINT_HEALTH_CHECK_INTERVAL")
        
        if env_value is None:
            return self.DEFAULT_ENDPOINT_HEALTH_CHECK_INTERVAL
        
        try:
            interval = int(env_value)
            if interval <= 0:
                logger.error(
                    f"Invalid GPU_ENDPOINT_HEALTH_CHECK_INTERVAL: {interval}. Must be positive. "
                    f"Using default: {self.DEFAULT_ENDPOINT_HEALTH_CHECK_INTERVAL}"
                )
                return self.DEFAULT_ENDPOINT_HEALTH_CHECK_INTERVAL
            
            logger.info(f"Loaded endpoint health check interval: {interval}s")
            return interval
            
        except ValueError:
            logger.error(
                f"Invalid GPU_ENDPOINT_HEALTH_CHECK_INTERVAL format: '{env_value}'. "
                f"Using default: {self.DEFAULT_ENDPOINT_HEALTH_CHECK_INTERVAL}"
            )
            return self.DEFAULT_ENDPOINT_HEALTH_CHECK_INTERVAL
    
    @staticmethod
    def _is_valid_threshold(value: float) -> bool:
        """Validate that threshold value is between 0 and 100."""
        return 0.0 <= value <= 100.0