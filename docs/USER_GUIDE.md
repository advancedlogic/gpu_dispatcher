# GPU Worker Pool User Guide

This guide provides comprehensive information on how to use the GPU Worker Pool system effectively in different scenarios.

## Table of Contents

- [Getting Started](#getting-started)
- [Basic Usage Patterns](#basic-usage-patterns)
- [Configuration Guide](#configuration-guide)
- [Advanced Usage](#advanced-usage)
- [Error Handling](#error-handling)
- [Monitoring and Debugging](#monitoring-and-debugging)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Access to a GPU statistics service
- Network connectivity to the GPU service endpoint

### Installation

```bash
pip install gpu-worker-pool
```

### Quick Start

The simplest way to use the GPU Worker Pool is with the context manager pattern:

```python
import asyncio
from gpu_worker_pool.client import GPUWorkerPoolClient

async def main():
    async with GPUWorkerPoolClient() as client:
        # Request a GPU
        assignment = await client.request_gpu()
        print(f"Got GPU {assignment.gpu_id}")
        
        # Do your GPU work here
        await asyncio.sleep(2.0)  # Simulate work
        
        # Release the GPU
        await client.release_gpu(assignment)
        print("GPU released")

asyncio.run(main())
```

## Basic Usage Patterns

### Pattern 1: Manual Resource Management

```python
import asyncio
from gpu_worker_pool.client import GPUWorkerPoolClient

async def manual_management():
    client = GPUWorkerPoolClient()
    
    try:
        await client.start()
        
        assignment = await client.request_gpu()
        print(f"Using GPU {assignment.gpu_id}")
        
        # Your GPU computation here
        await asyncio.sleep(1.0)
        
        await client.release_gpu(assignment)
        
    finally:
        await client.stop()

asyncio.run(manual_management())
```

### Pattern 2: Context Manager (Recommended)

```python
import asyncio
from gpu_worker_pool.client import GPUWorkerPoolClient

async def context_manager_usage():
    async with GPUWorkerPoolClient() as client:
        assignment = await client.request_gpu()
        
        # Your GPU computation here
        await asyncio.sleep(1.0)
        
        await client.release_gpu(assignment)

asyncio.run(context_manager_usage())
```

### Pattern 3: Automatic GPU Management

```python
import asyncio
from gpu_worker_pool.client import GPUWorkerPoolClient, GPUContextManager

async def automatic_gpu_management():
    async with GPUWorkerPoolClient() as client:
        async with GPUContextManager(client) as gpu_id:
            print(f"Automatically assigned GPU {gpu_id}")
            
            # Your GPU computation here
            await asyncio.sleep(1.0)
            
            # GPU is automatically released when exiting context

asyncio.run(automatic_gpu_management())
```

### Pattern 4: Factory Function

```python
import asyncio
from gpu_worker_pool.client import gpu_worker_pool_client, GPUContextManager

async def factory_function_usage():
    async with gpu_worker_pool_client(memory_threshold=75.0) as client:
        async with GPUContextManager(client) as gpu_id:
            print(f"Using GPU {gpu_id} with custom config")
            await asyncio.sleep(1.0)

asyncio.run(factory_function_usage())
```

## Configuration Guide

### Environment Variable Configuration

The most flexible way to configure the GPU Worker Pool is through environment variables:

```bash
# Set environment variables
export GPU_SERVICE_ENDPOINT="https://gpu-service.company.com"
export GPU_MEMORY_THRESHOLD_PERCENT="75.0"
export GPU_UTILIZATION_THRESHOLD_PERCENT="85.0"
export GPU_POLLING_INTERVAL="3"
```

```python
# Use with environment variables
async with GPUWorkerPoolClient() as client:
    # Configuration is automatically loaded from environment
    pass
```

### Programmatic Configuration

Override environment variables with programmatic configuration:

```python
client = GPUWorkerPoolClient(
    service_endpoint="http://localhost:8080",
    memory_threshold=70.0,
    utilization_threshold=80.0,
    worker_timeout=600.0,
    request_timeout=15.0,
    polling_interval=5
)
```

### Configuration for Different Environments

#### Development Environment

```python
dev_config = {
    "service_endpoint": "http://localhost:8080",
    "memory_threshold": 75.0,
    "utilization_threshold": 85.0,
    "worker_timeout": 60.0,
    "polling_interval": 2
}

async with gpu_worker_pool_client(**dev_config) as client:
    # Development usage
    pass
```

#### Production Environment

```python
prod_config = {
    "service_endpoint": "https://gpu-service.company.com",
    "memory_threshold": 70.0,
    "utilization_threshold": 80.0,
    "worker_timeout": 600.0,
    "polling_interval": 5
}

async with gpu_worker_pool_client(**prod_config) as client:
    # Production usage
    pass
```

#### High-Throughput Batch Processing

```python
batch_config = {
    "memory_threshold": 85.0,
    "utilization_threshold": 90.0,
    "worker_timeout": 1800.0,  # 30 minutes
    "polling_interval": 10
}

async with gpu_worker_pool_client(**batch_config) as client:
    # Batch processing usage
    pass
```

### Configuration Validation

Validate your configuration before using it:

```python
from examples.config_examples import ConfigurationValidator

def validate_config(memory_threshold, utilization_threshold, endpoint):
    validator = ConfigurationValidator()
    
    # Validate thresholds
    threshold_result = validator.validate_thresholds(
        memory_threshold, utilization_threshold
    )
    
    # Validate endpoint
    endpoint_result = validator.validate_service_endpoint(endpoint)
    
    if not threshold_result["valid"]:
        print(f"Threshold errors: {threshold_result['errors']}")
        return False
    
    if not endpoint_result["valid"]:
        print(f"Endpoint errors: {endpoint_result['errors']}")
        return False
    
    # Print warnings and recommendations
    if threshold_result["warnings"]:
        print(f"Warnings: {threshold_result['warnings']}")
    
    if threshold_result["recommendations"]:
        print(f"Recommendations: {threshold_result['recommendations']}")
    
    return True

# Validate before using
if validate_config(75.0, 85.0, "https://gpu-service.com"):
    # Use configuration
    pass
```

## Advanced Usage

### Multiple Concurrent Workers

```python
import asyncio
from gpu_worker_pool.client import GPUWorkerPoolClient, GPUContextManager

async def worker_task(client, worker_id, duration):
    """Simulate a worker task."""
    try:
        async with GPUContextManager(client, timeout=120.0) as gpu_id:
            print(f"Worker {worker_id} got GPU {gpu_id}")
            await asyncio.sleep(duration)
            print(f"Worker {worker_id} finished")
            return f"worker-{worker_id}-success"
    except Exception as e:
        print(f"Worker {worker_id} failed: {e}")
        return f"worker-{worker_id}-failed"

async def run_multiple_workers():
    async with GPUWorkerPoolClient() as client:
        # Create multiple worker tasks
        tasks = [
            worker_task(client, i, 1.0 + (i * 0.5))
            for i in range(5)
        ]
        
        # Run workers concurrently
        results = await asyncio.gather(*tasks)
        
        # Analyze results
        successes = [r for r in results if r.endswith('-success')]
        failures = [r for r in results if r.endswith('-failed')]
        
        print(f"Completed: {len(successes)} successes, {len(failures)} failures")

asyncio.run(run_multiple_workers())
```

### Machine Learning Training Pipeline

```python
import asyncio
from gpu_worker_pool.client import gpu_worker_pool_client, GPUContextManager

async def train_model(client, model_name, epochs, batch_size):
    """Simulate model training."""
    async with GPUContextManager(client, timeout=1800.0) as gpu_id:
        print(f"Training {model_name} on GPU {gpu_id}")
        print(f"Config: {epochs} epochs, batch size {batch_size}")
        
        # Simulate training epochs
        for epoch in range(epochs):
            await asyncio.sleep(0.5)  # Simulate epoch training
            print(f"{model_name} epoch {epoch + 1}/{epochs} complete")
        
        print(f"{model_name} training complete on GPU {gpu_id}")
        return {"model": model_name, "gpu": gpu_id, "epochs": epochs}

async def ml_pipeline():
    # Configuration for ML workloads
    ml_config = {
        "memory_threshold": 80.0,
        "utilization_threshold": 90.0,
        "worker_timeout": 1800.0,  # 30 minutes
        "polling_interval": 5
    }
    
    async with gpu_worker_pool_client(**ml_config) as client:
        # Define training jobs
        training_jobs = [
            train_model(client, "ResNet50", 10, 32),
            train_model(client, "BERT", 5, 16),
            train_model(client, "GPT", 8, 8)
        ]
        
        # Run training jobs
        results = await asyncio.gather(*training_jobs)
        
        print("Training pipeline complete:")
        for result in results:
            print(f"  {result['model']}: {result['epochs']} epochs on GPU {result['gpu']}")

asyncio.run(ml_pipeline())
```

### Resource Monitoring and Scaling

```python
import asyncio
from gpu_worker_pool.client import GPUWorkerPoolClient

async def monitor_and_scale():
    async with GPUWorkerPoolClient() as client:
        while True:
            status = client.get_pool_status()
            metrics = client.get_detailed_metrics()
            
            # Calculate utilization
            if status.total_gpus > 0:
                utilization = (status.total_gpus - status.available_gpus) / status.total_gpus
                
                print(f"Pool Status:")
                print(f"  Total GPUs: {status.total_gpus}")
                print(f"  Available GPUs: {status.available_gpus}")
                print(f"  Utilization: {utilization:.1%}")
                print(f"  Active workers: {status.active_workers}")
                print(f"  Blocked workers: {status.blocked_workers}")
                
                # Scaling decisions
                if utilization > 0.9 and status.blocked_workers > 0:
                    print("⚠️  High utilization with blocked workers - consider scaling")
                elif utilization < 0.3:
                    print("ℹ️  Low utilization - consider scaling down")
                else:
                    print("✅ Utilization is optimal")
            
            await asyncio.sleep(10)  # Monitor every 10 seconds

# Run monitoring (would typically run in background)
# asyncio.run(monitor_and_scale())
```

### Custom Timeout Handling

```python
import asyncio
from gpu_worker_pool.client import GPUWorkerPoolClient
from gpu_worker_pool.worker_pool_manager import WorkerTimeoutError

async def adaptive_timeout_example():
    async with GPUWorkerPoolClient() as client:
        # Check current pool status to determine timeout
        status = client.get_pool_status()
        
        if status.blocked_workers > status.total_gpus:
            # High contention - use longer timeout
            timeout = 600.0  # 10 minutes
            print("High contention detected, using extended timeout")
        else:
            # Normal conditions - use shorter timeout
            timeout = 120.0  # 2 minutes
            print("Normal conditions, using standard timeout")
        
        try:
            assignment = await client.request_gpu(timeout=timeout)
            print(f"Got GPU {assignment.gpu_id}")
            
            # Do work
            await asyncio.sleep(2.0)
            
            await client.release_gpu(assignment)
            
        except WorkerTimeoutError:
            print(f"Request timed out after {timeout} seconds")
            # Implement fallback logic
            print("Deferring work to later...")

asyncio.run(adaptive_timeout_example())
```

## Error Handling

### Common Error Scenarios

#### Timeout Errors

```python
import asyncio
from gpu_worker_pool.client import GPUWorkerPoolClient
from gpu_worker_pool.worker_pool_manager import WorkerTimeoutError

async def handle_timeouts():
    async with GPUWorkerPoolClient() as client:
        try:
            assignment = await client.request_gpu(timeout=60.0)
            await client.release_gpu(assignment)
            
        except WorkerTimeoutError as e:
            print(f"GPU request timed out: {e}")
            # Implement retry logic or fallback
            await asyncio.sleep(30)  # Wait before retry
            
        except asyncio.TimeoutError:
            print("Operation timed out at a lower level")

asyncio.run(handle_timeouts())
```

#### Service Unavailable

```python
from gpu_worker_pool.client import GPUWorkerPoolClient
from gpu_worker_pool.http_client import ServiceUnavailableError

async def handle_service_unavailable():
    try:
        async with GPUWorkerPoolClient() as client:
            assignment = await client.request_gpu()
            await client.release_gpu(assignment)
            
    except ServiceUnavailableError as e:
        print(f"GPU service is unavailable: {e}")
        # Implement fallback or retry logic
        
    except Exception as e:
        print(f"Unexpected error: {e}")

asyncio.run(handle_service_unavailable())
```

#### Invalid Configuration

```python
from gpu_worker_pool.client import GPUWorkerPoolClient

async def handle_invalid_config():
    try:
        # This will use default values for invalid thresholds
        client = GPUWorkerPoolClient(
            memory_threshold=150.0,  # Invalid - over 100%
            utilization_threshold=-10.0  # Invalid - negative
        )
        
        async with client:
            # Client will use default values
            status = client.get_pool_status()
            print(f"Using defaults: {status}")
            
    except Exception as e:
        print(f"Configuration error: {e}")

asyncio.run(handle_invalid_config())
```

### Retry Patterns

#### Exponential Backoff

```python
import asyncio
import random
from gpu_worker_pool.client import GPUWorkerPoolClient
from gpu_worker_pool.worker_pool_manager import WorkerTimeoutError

async def exponential_backoff_retry(client, max_retries=3):
    """Retry GPU request with exponential backoff."""
    for attempt in range(max_retries):
        try:
            assignment = await client.request_gpu(timeout=60.0)
            return assignment
            
        except WorkerTimeoutError:
            if attempt == max_retries - 1:
                raise  # Re-raise on final attempt
            
            # Exponential backoff with jitter
            delay = (2 ** attempt) + random.uniform(0, 1)
            print(f"Attempt {attempt + 1} failed, retrying in {delay:.1f}s")
            await asyncio.sleep(delay)

async def retry_example():
    async with GPUWorkerPoolClient() as client:
        try:
            assignment = await exponential_backoff_retry(client)
            print(f"Got GPU {assignment.gpu_id} after retries")
            await client.release_gpu(assignment)
            
        except WorkerTimeoutError:
            print("All retry attempts failed")

asyncio.run(retry_example())
```

#### Circuit Breaker Pattern

```python
import asyncio
import time
from gpu_worker_pool.client import GPUWorkerPoolClient
from gpu_worker_pool.worker_pool_manager import WorkerTimeoutError

class CircuitBreaker:
    def __init__(self, failure_threshold=5, recovery_timeout=60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    
    def can_execute(self):
        if self.state == "CLOSED":
            return True
        elif self.state == "OPEN":
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "HALF_OPEN"
                return True
            return False
        else:  # HALF_OPEN
            return True
    
    def on_success(self):
        self.failure_count = 0
        self.state = "CLOSED"
    
    def on_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"

async def circuit_breaker_example():
    circuit_breaker = CircuitBreaker()
    
    async with GPUWorkerPoolClient() as client:
        for i in range(10):
            if not circuit_breaker.can_execute():
                print(f"Request {i}: Circuit breaker is OPEN, skipping")
                await asyncio.sleep(1)
                continue
            
            try:
                assignment = await client.request_gpu(timeout=30.0)
                print(f"Request {i}: Got GPU {assignment.gpu_id}")
                circuit_breaker.on_success()
                await client.release_gpu(assignment)
                
            except WorkerTimeoutError:
                print(f"Request {i}: Timeout")
                circuit_breaker.on_failure()
            
            await asyncio.sleep(1)

asyncio.run(circuit_breaker_example())
```

## Monitoring and Debugging

### Pool Status Monitoring

```python
import asyncio
from gpu_worker_pool.client import GPUWorkerPoolClient

async def monitor_pool_status():
    async with GPUWorkerPoolClient() as client:
        status = client.get_pool_status()
        
        print("=== Pool Status ===")
        print(f"Total GPUs: {status.total_gpus}")
        print(f"Available GPUs: {status.available_gpus}")
        print(f"Active workers: {status.active_workers}")
        print(f"Blocked workers: {status.blocked_workers}")
        
        # Calculate metrics
        if status.total_gpus > 0:
            utilization = (status.total_gpus - status.available_gpus) / status.total_gpus
            print(f"Utilization: {utilization:.1%}")
        
        # Check for issues
        if status.available_gpus == 0 and status.blocked_workers > 0:
            print("⚠️  All GPUs are busy, workers are waiting")
        
        if status.blocked_workers > status.total_gpus * 2:
            print("⚠️  High number of blocked workers, consider scaling")

asyncio.run(monitor_pool_status())
```

### Detailed Metrics

```python
import asyncio
import json
from gpu_worker_pool.client import GPUWorkerPoolClient

async def detailed_metrics():
    async with GPUWorkerPoolClient() as client:
        metrics = client.get_detailed_metrics()
        
        print("=== Detailed Metrics ===")
        print(f"Timestamp: {metrics['timestamp']}")
        print(f"Is running: {metrics['is_running']}")
        
        # GPU-level metrics
        print("\n--- GPU Metrics ---")
        for gpu in metrics['gpu_metrics']:
            print(f"GPU {gpu['gpu_id']} ({gpu['name']}):")
            print(f"  Memory: {gpu['memory_usage_percent']:.1f}%")
            print(f"  Utilization: {gpu['utilization_percent']:.1f}%")
            print(f"  Available: {gpu['is_available']}")
            print(f"  Assigned workers: {gpu['assigned_workers']}")
        
        # Thresholds
        print("\n--- Thresholds ---")
        thresholds = metrics['thresholds']
        print(f"Memory threshold: {thresholds['memory_threshold_percent']}%")
        print(f"Utilization threshold: {thresholds['utilization_threshold_percent']}%")
        
        # Worker assignments
        print("\n--- Active Assignments ---")
        for assignment in metrics['assignment_metrics']:
            print(f"Worker {assignment['worker_id']}: GPU {assignment['gpu_id']} "
                  f"({assignment['duration_seconds']:.1f}s)")

asyncio.run(detailed_metrics())
```

### Logging Configuration

```python
import logging
import asyncio
from gpu_worker_pool.client import GPUWorkerPoolClient

# Configure detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Enable debug logging for specific components
logging.getLogger('gpu_worker_pool.client').setLevel(logging.DEBUG)
logging.getLogger('gpu_worker_pool.worker_pool_manager').setLevel(logging.INFO)
logging.getLogger('gpu_worker_pool.gpu_monitor').setLevel(logging.WARNING)

async def logging_example():
    async with GPUWorkerPoolClient() as client:
        assignment = await client.request_gpu()
        await asyncio.sleep(1.0)
        await client.release_gpu(assignment)

asyncio.run(logging_example())
```

### Performance Profiling

```python
import asyncio
import time
from gpu_worker_pool.client import GPUWorkerPoolClient, GPUContextManager

async def profile_performance():
    async with GPUWorkerPoolClient() as client:
        # Measure request latency
        start_time = time.time()
        assignment = await client.request_gpu()
        request_latency = time.time() - start_time
        
        print(f"GPU request latency: {request_latency:.3f}s")
        
        # Measure release latency
        start_time = time.time()
        await client.release_gpu(assignment)
        release_latency = time.time() - start_time
        
        print(f"GPU release latency: {release_latency:.3f}s")

async def profile_context_manager():
    async with GPUWorkerPoolClient() as client:
        start_time = time.time()
        
        async with GPUContextManager(client) as gpu_id:
            context_setup_time = time.time() - start_time
            print(f"Context manager setup: {context_setup_time:.3f}s")
            await asyncio.sleep(0.1)  # Simulate work
        
        total_time = time.time() - start_time
        print(f"Total context manager time: {total_time:.3f}s")

# Run profiling
asyncio.run(profile_performance())
asyncio.run(profile_context_manager())
```

## Best Practices

### 1. Always Use Context Managers

```python
# ✅ Good - Automatic resource cleanup
async with GPUWorkerPoolClient() as client:
    async with GPUContextManager(client) as gpu_id:
        # Use GPU
        pass

# ❌ Avoid - Manual resource management
client = GPUWorkerPoolClient()
await client.start()
assignment = await client.request_gpu()
# ... easy to forget cleanup
await client.release_gpu(assignment)
await client.stop()
```

### 2. Configure Appropriate Timeouts

```python
# ✅ Good - Set timeouts based on workload
async with GPUWorkerPoolClient(worker_timeout=300.0) as client:
    # Short timeout for interactive work
    assignment = await client.request_gpu(timeout=60.0)
    
    # Long timeout for batch processing
    batch_assignment = await client.request_gpu(timeout=1800.0)
```

### 3. Handle Errors Gracefully

```python
# ✅ Good - Comprehensive error handling
try:
    async with GPUWorkerPoolClient() as client:
        assignment = await client.request_gpu(timeout=120.0)
        # Use GPU
        await client.release_gpu(assignment)
        
except WorkerTimeoutError:
    print("No GPU available, deferring work")
except ServiceUnavailableError:
    print("Service unavailable, will retry later")
except Exception as e:
    print(f"Unexpected error: {e}")
```

### 4. Monitor Pool Health

```python
# ✅ Good - Regular health monitoring
async def health_check(client):
    status = client.get_pool_status()
    
    if status.total_gpus == 0:
        return "No GPUs available"
    
    utilization = (status.total_gpus - status.available_gpus) / status.total_gpus
    
    if utilization > 0.9:
        return "High utilization"
    elif status.blocked_workers > status.total_gpus:
        return "High contention"
    else:
        return "Healthy"
```

### 5. Use Environment Variables for Configuration

```python
# ✅ Good - Environment-based configuration
# Set in environment or container
export GPU_MEMORY_THRESHOLD_PERCENT="75.0"
export GPU_UTILIZATION_THRESHOLD_PERCENT="85.0"

# Use defaults from environment
async with GPUWorkerPoolClient() as client:
    pass
```

### 6. Implement Proper Logging

```python
import logging

# ✅ Good - Structured logging
logger = logging.getLogger(__name__)

async def my_gpu_task():
    async with GPUWorkerPoolClient() as client:
        try:
            assignment = await client.request_gpu()
            logger.info(f"Started task on GPU {assignment.gpu_id}")
            
            # Do work
            await asyncio.sleep(1.0)
            
            await client.release_gpu(assignment)
            logger.info(f"Completed task on GPU {assignment.gpu_id}")
            
        except Exception as e:
            logger.error(f"Task failed: {e}")
            raise
```

## Troubleshooting

### Common Issues

#### Issue: "Client must be started before requesting GPU"

**Cause:** Trying to use client methods before calling `start()` or outside context manager.

**Solution:**
```python
# ✅ Use context manager
async with GPUWorkerPoolClient() as client:
    assignment = await client.request_gpu()

# ✅ Or manual start/stop
client = GPUWorkerPoolClient()
await client.start()
try:
    assignment = await client.request_gpu()
finally:
    await client.stop()
```

#### Issue: "Worker request timed out"

**Cause:** No GPUs available within the timeout period.

**Solutions:**
1. Increase timeout:
```python
assignment = await client.request_gpu(timeout=600.0)  # 10 minutes
```

2. Check pool status:
```python
status = client.get_pool_status()
if status.available_gpus == 0:
    print("No GPUs available, try again later")
```

3. Adjust thresholds:
```python
client = GPUWorkerPoolClient(
    memory_threshold=85.0,  # More permissive
    utilization_threshold=90.0
)
```

#### Issue: "Service unavailable"

**Cause:** GPU statistics service is down or unreachable.

**Solutions:**
1. Check service endpoint:
```python
client = GPUWorkerPoolClient(
    service_endpoint="http://correct-endpoint:8080"
)
```

2. Verify network connectivity
3. Check service logs
4. Implement retry logic

#### Issue: High memory usage or slow performance

**Cause:** Too many concurrent workers or inefficient resource usage.

**Solutions:**
1. Limit concurrent workers:
```python
semaphore = asyncio.Semaphore(4)  # Max 4 concurrent workers

async def limited_worker(client):
    async with semaphore:
        async with GPUContextManager(client) as gpu_id:
            # Do work
            pass
```

2. Monitor resource usage:
```python
metrics = client.get_detailed_metrics()
for gpu in metrics['gpu_metrics']:
    if gpu['memory_usage_percent'] > 90:
        print(f"GPU {gpu['gpu_id']} high memory usage")
```

### Debugging Steps

1. **Enable Debug Logging:**
```python
import logging
logging.getLogger('gpu_worker_pool').setLevel(logging.DEBUG)
```

2. **Check Pool Status:**
```python
status = client.get_pool_status()
print(f"Status: {status}")
```

3. **Verify Configuration:**
```python
from gpu_worker_pool.config import EnvironmentConfigurationManager
config = EnvironmentConfigurationManager()
print(f"Endpoint: {config.get_service_endpoint()}")
print(f"Thresholds: {config.get_memory_threshold()}%, {config.get_utilization_threshold()}%")
```

4. **Test Service Connectivity:**
```python
import aiohttp

async def test_service():
    async with aiohttp.ClientSession() as session:
        async with session.get("http://gpu-service:8080/summary") as response:
            print(f"Status: {response.status}")
            data = await response.json()
            print(f"Response: {data}")
```

5. **Monitor Resource Usage:**
```python
import psutil
import asyncio

async def monitor_resources():
    while True:
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        print(f"CPU: {cpu_percent}%, Memory: {memory.percent}%")
        await asyncio.sleep(5)
```

### Performance Tuning

1. **Adjust Polling Interval:**
```python
# Faster response, higher overhead
client = GPUWorkerPoolClient(polling_interval=1)

# Slower response, lower overhead
client = GPUWorkerPoolClient(polling_interval=10)
```

2. **Optimize Thresholds:**
```python
# Conservative - stable but lower utilization
client = GPUWorkerPoolClient(
    memory_threshold=70.0,
    utilization_threshold=80.0
)

# Aggressive - higher utilization but less stable
client = GPUWorkerPoolClient(
    memory_threshold=85.0,
    utilization_threshold=95.0
)
```

3. **Tune Timeouts:**
```python
# Short timeout for interactive workloads
client = GPUWorkerPoolClient(
    worker_timeout=60.0,
    request_timeout=5.0
)

# Long timeout for batch processing
client = GPUWorkerPoolClient(
    worker_timeout=1800.0,
    request_timeout=30.0
)
```

This user guide should help you effectively use the GPU Worker Pool system in various scenarios. For more specific examples, see the `examples/` directory in the repository.