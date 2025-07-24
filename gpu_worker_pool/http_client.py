"""HTTP client for GPU statistics service."""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import aiohttp
from .models import GPUStats, create_gpu_stats_from_json

logger = logging.getLogger(__name__)


class ServiceUnavailableError(Exception):
    """Raised when the GPU service is unavailable after retries."""
    pass


class RetryableError(Exception):
    """Base class for errors that should trigger retry logic."""
    pass


class NetworkRetryableError(RetryableError):
    """Network-related error that should be retried."""
    pass


class ServiceRetryableError(RetryableError):
    """Service-related error that should be retried."""
    pass


class GPUStatsHTTPClient(ABC):
    """Abstract base class for GPU statistics HTTP client."""
    
    @abstractmethod
    async def fetch_gpu_stats(self) -> Optional[GPUStats]:
        """Fetch GPU statistics from the service."""
        pass
    
    @abstractmethod
    async def close(self) -> None:
        """Close the HTTP client and cleanup resources."""
        pass


class AsyncGPUStatsHTTPClient(GPUStatsHTTPClient):
    """Async HTTP client for fetching GPU statistics with retry logic."""
    
    def __init__(self, 
                 endpoint: str, 
                 timeout: float = 10.0,
                 max_retries: int = 3,
                 base_retry_delay: float = 1.0,
                 max_retry_delay: float = 30.0,
                 backoff_multiplier: float = 2.0):
        """Initialize the HTTP client.
        
        Args:
            endpoint: Base URL of the GPU statistics service
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
            base_retry_delay: Base delay between retries in seconds
            max_retry_delay: Maximum delay between retries in seconds
            backoff_multiplier: Multiplier for exponential backoff
        """
        self.endpoint = endpoint.rstrip('/')
        self.timeout = timeout
        self.max_retries = max_retries
        self.base_retry_delay = base_retry_delay
        self.max_retry_delay = max_retry_delay
        self.backoff_multiplier = backoff_multiplier
        self._session: Optional[aiohttp.ClientSession] = None
        
        # Circuit breaker state
        self._consecutive_failures = 0
        self._circuit_breaker_threshold = 5
        self._circuit_breaker_reset_time = 60.0
        self._last_failure_time: Optional[float] = None
        
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session."""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session
    
    async def fetch_gpu_stats(self) -> Optional[GPUStats]:
        """Fetch GPU statistics with retry logic and circuit breaker.
        
        Returns:
            GPUStats object if successful, None if service unavailable
            
        Raises:
            ServiceUnavailableError: When service is unavailable after all retries
            ValueError: For invalid response format that shouldn't be retried
        """
        # Check circuit breaker
        if self._is_circuit_breaker_open():
            logger.warning("Circuit breaker is open, skipping GPU stats fetch")
            raise ServiceUnavailableError("Circuit breaker is open - service unavailable")
        
        url = f"{self.endpoint}/gpu/summary"
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                logger.debug(f"Fetching GPU stats from {url} (attempt {attempt + 1}/{self.max_retries + 1})")
                
                session = await self._get_session()
                async with session.get(url) as response:
                    if response.status == 200:
                        # Success - reset failure tracking
                        self._on_success()
                        
                        # Parse JSON response
                        try:
                            data = await response.json()
                            logger.debug(f"Received GPU stats: {len(data.get('gpus_summary', []))} GPUs")
                            
                            # Validate and create GPUStats object
                            gpu_stats = create_gpu_stats_from_json(data)
                            return gpu_stats
                            
                        except (ValueError, KeyError) as e:
                            logger.error(f"Invalid JSON response format from {url}: {e}")
                            # Don't retry for invalid response format
                            raise ValueError(f"Invalid response format: {e}")
                    
                    elif response.status >= 500:
                        # Server error - retryable
                        error_msg = f"HTTP {response.status} server error from {url}"
                        logger.warning(f"{error_msg} (attempt {attempt + 1})")
                        last_exception = ServiceRetryableError(error_msg)
                        
                    elif response.status == 429:
                        # Rate limited - retryable
                        error_msg = f"Rate limited by {url}"
                        logger.warning(f"{error_msg} (attempt {attempt + 1})")
                        last_exception = ServiceRetryableError(error_msg)
                        
                    else:
                        # Client error (4xx) - not retryable
                        error_msg = f"HTTP {response.status} client error from {url}"
                        logger.error(error_msg)
                        self._on_failure()
                        raise aiohttp.ClientResponseError(
                            request_info=response.request_info,
                            history=response.history,
                            status=response.status
                        )
                        
            except aiohttp.ClientConnectorError as e:
                # Connection error - retryable
                error_msg = f"Connection error to {url}: {e}"
                logger.warning(f"{error_msg} (attempt {attempt + 1})")
                last_exception = NetworkRetryableError(error_msg)
                
            except aiohttp.ServerTimeoutError as e:
                # Timeout - retryable
                error_msg = f"Timeout connecting to {url}: {e}"
                logger.warning(f"{error_msg} (attempt {attempt + 1})")
                last_exception = NetworkRetryableError(error_msg)
                
            except asyncio.TimeoutError as e:
                # Asyncio timeout - retryable
                error_msg = f"Request timeout to {url}: {e}"
                logger.warning(f"{error_msg} (attempt {attempt + 1})")
                last_exception = NetworkRetryableError(error_msg)
                
            except aiohttp.ClientError as e:
                # Other client errors - retryable
                error_msg = f"Client error fetching from {url}: {e}"
                logger.warning(f"{error_msg} (attempt {attempt + 1})")
                last_exception = NetworkRetryableError(error_msg)
                
            except Exception as e:
                # Unexpected error - log and don't retry
                logger.error(f"Unexpected error fetching GPU stats from {url}: {e}", exc_info=True)
                self._on_failure()
                raise
            
            # If this wasn't the last attempt, wait before retrying
            if attempt < self.max_retries:
                retry_delay = self._calculate_retry_delay(attempt)
                logger.info(f"Retrying in {retry_delay:.1f} seconds...")
                await asyncio.sleep(retry_delay)
        
        # All retries exhausted
        self._on_failure()
        error_msg = f"Failed to fetch GPU stats after {self.max_retries + 1} attempts"
        if last_exception:
            error_msg += f": {last_exception}"
        
        logger.error(error_msg)
        raise ServiceUnavailableError(error_msg)
    
    def _is_circuit_breaker_open(self) -> bool:
        """Check if circuit breaker is open."""
        if self._consecutive_failures < self._circuit_breaker_threshold:
            return False
        
        if self._last_failure_time is None:
            return False
        
        # Check if enough time has passed to reset the circuit breaker
        import time
        time_since_failure = time.time() - self._last_failure_time
        if time_since_failure >= self._circuit_breaker_reset_time:
            logger.info("Circuit breaker reset - attempting to reconnect to service")
            self._consecutive_failures = 0
            self._last_failure_time = None
            return False
        
        return True
    
    def _on_success(self) -> None:
        """Handle successful request."""
        if self._consecutive_failures > 0:
            logger.info(f"Service recovered after {self._consecutive_failures} consecutive failures")
            self._consecutive_failures = 0
            self._last_failure_time = None
    
    def _on_failure(self) -> None:
        """Handle failed request."""
        import time
        self._consecutive_failures += 1
        self._last_failure_time = time.time()
        
        if self._consecutive_failures >= self._circuit_breaker_threshold:
            logger.warning(
                f"Circuit breaker opened after {self._consecutive_failures} consecutive failures. "
                f"Will retry after {self._circuit_breaker_reset_time} seconds."
            )
    
    def _calculate_retry_delay(self, attempt: int) -> float:
        """Calculate retry delay using exponential backoff."""
        delay = self.base_retry_delay * (self.backoff_multiplier ** attempt)
        return min(delay, self.max_retry_delay)
    
    def get_client_status(self) -> Dict[str, Any]:
        """Get current client status for monitoring."""
        return {
            "consecutive_failures": self._consecutive_failures,
            "circuit_breaker_open": self._is_circuit_breaker_open(),
            "last_failure_time": self._last_failure_time,
            "endpoint": self.endpoint,
            "timeout": self.timeout,
            "max_retries": self.max_retries
        }
    
    async def close(self) -> None:
        """Close the HTTP session and cleanup resources."""
        if self._session and not self._session.closed:
            await self._session.close()
            logger.debug("HTTP client session closed")


class MockGPUStatsHTTPClient(GPUStatsHTTPClient):
    """Mock HTTP client for testing purposes."""
    
    def __init__(self, mock_response: Optional[Dict[str, Any]] = None, 
                 should_fail: bool = False, 
                 failure_exception: Optional[Exception] = None):
        """Initialize mock client.
        
        Args:
            mock_response: Mock JSON response data
            should_fail: Whether to simulate failures
            failure_exception: Exception to raise on failure
        """
        self.mock_response = mock_response
        self.should_fail = should_fail
        self.failure_exception = failure_exception or aiohttp.ClientError("Mock failure")
        self.call_count = 0
        
    async def fetch_gpu_stats(self) -> Optional[GPUStats]:
        """Mock fetch GPU statistics."""
        self.call_count += 1
        
        if self.should_fail:
            raise self.failure_exception
            
        if self.mock_response is None:
            return None
            
        return create_gpu_stats_from_json(self.mock_response)
    
    async def close(self) -> None:
        """Mock close method."""
        pass