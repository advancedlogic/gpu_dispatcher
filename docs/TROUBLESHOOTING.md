# Troubleshooting Guide

This guide helps diagnose and resolve common issues with the GPU Worker Pool, with special focus on multi-endpoint configurations.

## Table of Contents

- [General Issues](#general-issues)
- [Multi-Endpoint Issues](#multi-endpoint-issues)
- [Network and Connectivity](#network-and-connectivity)
- [Performance Issues](#performance-issues)
- [Configuration Problems](#configuration-problems)
- [Debugging Tools](#debugging-tools)

## General Issues

### GPU Not Being Assigned

**Symptoms:**
- `request_gpu()` times out
- Workers remain in blocked state
- No GPUs show as available

**Possible Causes & Solutions:**

1. **All GPUs exceed thresholds**
   ```python
   # Check current thresholds
   metrics = client.get_detailed_metrics()
   print(f"Memory threshold: {metrics['thresholds']['memory_threshold_percent']}%")
   print(f"Utilization threshold: {metrics['thresholds']['utilization_threshold_percent']}%")
   
   # Check GPU states
   for gpu in metrics['gpu_metrics']:
       print(f"GPU {gpu['gpu_id']}: {gpu['memory_usage_percent']}% memory, {gpu['utilization_percent']}% util")
   ```
   
   **Solution**: Lower thresholds or wait for GPUs to become available

2. **GPU statistics service is down**
   ```python
   # Check service health
   try:
       status = client.get_pool_status()
   except Exception as e:
       print(f"Service unavailable: {e}")
   ```
   
   **Solution**: Ensure GPU statistics server is running

3. **Network connectivity issues**
   ```bash
   # Test connectivity
   curl http://gpu-server:8000/health
   ```

### Stale GPU Assignments

**Symptoms:**
- `StaleAssignmentError` when releasing GPU
- GPU remains marked as assigned after worker crash

**Solutions:**

1. **Implement proper cleanup**
   ```python
   async with GPUContextManager(client) as gpu_id:
       # GPU automatically released even if exception occurs
       await do_work(gpu_id)
   ```

2. **Handle worker timeouts**
   ```python
   client = GPUWorkerPoolClient(
       worker_timeout=300.0  # 5 minutes
   )
   ```

## Multi-Endpoint Issues

### Endpoints Not Detected

**Symptom:** Client runs in single-endpoint mode despite multiple endpoints configured, or `TypeError: got an unexpected keyword argument 'service_endpoints'`

**Diagnosis:**
```python
print(f"Multi-endpoint mode: {client.is_multi_endpoint_mode()}")
endpoints = client.get_endpoints_info()
print(f"Endpoints detected: {len(endpoints) if endpoints else 0}")
```

**Common Causes:**

1. **Using older package version (most common)**
   ```python
   # Check if multi-endpoint support is available
   import inspect
   from gpu_worker_pool.client import GPUWorkerPoolClient
   
   sig = inspect.signature(GPUWorkerPoolClient.__init__)
   if 'service_endpoints' not in sig.parameters:
       print("Multi-endpoint support not available!")
   ```

   **Solutions:**
   - Run with local code: `PYTHONPATH=. python your_script.py`
   - Install in development mode: `pip install -e .` 
   - Use the helper script: `python run_examples.py`

2. **Wrong parameter name**
   ```python
   # Wrong
   client = GPUWorkerPoolClient(
       service_endpoint="http://gpu1:8000,http://gpu2:8000"  # Should be plural!
   )
   
   # Correct
   client = GPUWorkerPoolClient(
       service_endpoints="http://gpu1:8000,http://gpu2:8000"
   )
   ```

3. **Wrong environment variable**
   ```bash
   # Wrong
   export GPU_SERVICE_ENDPOINTS="..."  # Missing 'STATS'
   
   # Correct
   export GPU_STATS_SERVICE_ENDPOINTS="..."
   ```

### Some Endpoints Unhealthy

**Symptom:** Not all configured endpoints are available

**Diagnosis:**
```python
# Check endpoint health
recovery_status = client.get_error_recovery_status()
print(f"Healthy: {recovery_status['endpoint_health_summary']['healthy_count']}")
print(f"Degraded: {recovery_status['endpoint_health_summary']['degraded_count']}")

# Detailed status
client.print_error_recovery_summary()
```

**Common Causes:**

1. **Server not running**
   ```bash
   # Check each endpoint
   for endpoint in gpu1 gpu2 gpu3; do
       echo "Checking $endpoint..."
       curl -f http://$endpoint:8000/health || echo "Failed"
   done
   ```

2. **Network issues**
   ```python
   # Check response times
   endpoints = client.get_endpoints_info()
   for ep in endpoints:
       if ep['response_time_ms'] > 1000:
           print(f"Slow endpoint: {ep['endpoint_id']} ({ep['response_time_ms']}ms)")
   ```

3. **Circuit breaker opened**
   ```python
   # Check circuit breaker states
   status = client.get_error_recovery_status()
   for endpoint_id, cb_stats in status['degradation_manager']['circuit_breaker_stats'].items():
       if cb_stats['state'] == 'open':
           print(f"Circuit breaker open for {endpoint_id}: {cb_stats['failure_rate']}% failures")
   ```

### Load Balancing Not Working as Expected

**Symptom:** Requests not distributed according to strategy

**Diagnosis:**
```python
# Check load balancer statistics
metrics = client.get_detailed_metrics()
lb_stats = metrics.get('load_balancer', {})
print(f"Strategy: {lb_stats.get('strategy_name')}")
print(f"Total requests: {lb_stats.get('total_requests')}")

# Check per-endpoint distribution
for endpoint_id, stats in lb_stats.get('endpoint_request_counts', {}).items():
    print(f"{endpoint_id}: {stats} requests")
```

**Solutions by Strategy:**

1. **Availability-based issues**
   - Verify endpoints have different availability levels
   - Check if all endpoints are at similar capacity

2. **Round-robin issues**
   - Ensure all endpoints are healthy
   - Check for request timing patterns

3. **Weighted issues**
   - Verify endpoints have different GPU counts
   - Check total GPU calculation

### Global GPU ID Issues

**Symptom:** Code breaks with "server1:0" format IDs

**Solutions:**

1. **Parse IDs when needed**
   ```python
   def parse_global_gpu_id(global_id):
       if ':' in str(global_id):
           endpoint_id, local_id = global_id.split(':', 1)
           return endpoint_id, int(local_id)
       return None, int(global_id)  # Single-endpoint mode
   ```

2. **Update logging/monitoring**
   ```python
   # Works with both ID formats
   logger.info(f"GPU assigned: {assignment.gpu_id}")
   ```

## Network and Connectivity

### Connection Timeouts

**Symptom:** Frequent timeout errors

**Solutions:**

1. **Increase timeouts**
   ```python
   client = GPUWorkerPoolClient(
       request_timeout=60.0  # Increase from default 30s
   )
   ```
   
   Or via environment:
   ```bash
   export GPU_ENDPOINT_TIMEOUT="60.0"
   ```

2. **Check network latency**
   ```bash
   # Ping each endpoint
   for endpoint in gpu1 gpu2 gpu3; do
       ping -c 4 $endpoint
   done
   ```

### SSL/TLS Issues

**Symptom:** HTTPS connections fail

**Solutions:**

1. **Verify certificates**
   ```bash
   curl -v https://gpu-server:8443/health
   ```

2. **Configure SSL properly**
   ```python
   # If using self-signed certificates
   import ssl
   ssl_context = ssl.create_default_context()
   ssl_context.check_hostname = False
   ssl_context.verify_mode = ssl.CERT_NONE
   ```

## Performance Issues

### Slow Response Times

**Symptom:** GPU requests take longer than expected

**Diagnosis:**
```python
import time

start = time.time()
assignment = await client.request_gpu()
duration = time.time() - start
print(f"Request took {duration:.2f} seconds")
```

**Solutions:**

1. **Reduce polling interval**
   ```python
   client = GPUWorkerPoolClient(
       polling_interval=2  # Poll every 2 seconds instead of 5
   )
   ```

2. **Check endpoint performance**
   ```python
   endpoints = client.get_endpoints_info()
   slow_endpoints = [ep for ep in endpoints if ep['response_time_ms'] > 500]
   ```

### High Memory Usage

**Symptom:** Client process uses excessive memory

**Solutions:**

1. **Limit concurrent requests**
   ```python
   # Use semaphore to limit concurrent GPU requests
   semaphore = asyncio.Semaphore(10)
   
   async def limited_gpu_request():
       async with semaphore:
           return await client.request_gpu()
   ```

2. **Monitor connection pools**
   ```python
   # Ensure connections are properly closed
   async with GPUWorkerPoolClient() as client:
       # Connections managed automatically
       pass
   ```

## Configuration Problems

### Environment Variables Not Working

**Common Issues:**

1. **Variable name typos**
   ```bash
   # Check current environment
   env | grep GPU_
   ```

2. **Override precedence**
   ```python
   # Code parameters override environment variables
   client = GPUWorkerPoolClient(
       memory_threshold=70.0  # This overrides GPU_MEMORY_THRESHOLD_PERCENT
   )
   ```

### Configuration Validation

**Validate configuration before deployment:**

```python
def validate_configuration(config):
    errors = []
    
    # Check thresholds
    if not 0 <= config.get('memory_threshold', 80) <= 100:
        errors.append("Memory threshold must be 0-100")
    
    # Check endpoints
    endpoints = config.get('service_endpoints', '').split(',')
    for endpoint in endpoints:
        if not endpoint.startswith(('http://', 'https://')):
            errors.append(f"Invalid endpoint URL: {endpoint}")
    
    return errors
```

## Debugging Tools

### Enable Debug Logging

```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)
logging.getLogger('gpu_worker_pool').setLevel(logging.DEBUG)
```

### Health Check Script

Create a health check script:

```python
#!/usr/bin/env python3
import asyncio
from gpu_worker_pool.client import GPUWorkerPoolClient

async def health_check():
    client = GPUWorkerPoolClient()
    
    try:
        await client.start()
        
        # Basic health
        status = client.get_pool_status()
        print(f"✓ Connected: {status.total_gpus} GPUs found")
        
        # Multi-endpoint health
        if client.is_multi_endpoint_mode():
            endpoints = client.get_endpoints_info()
            for ep in endpoints:
                symbol = "✓" if ep['is_healthy'] else "✗"
                print(f"{symbol} {ep['endpoint_id']}: {ep['url']}")
        
        # Try GPU assignment
        try:
            assignment = await client.request_gpu(timeout=10.0)
            await client.release_gpu(assignment)
            print("✓ GPU assignment working")
        except Exception as e:
            print(f"✗ GPU assignment failed: {e}")
            
    finally:
        await client.stop()

asyncio.run(health_check())
```

### Monitoring Script

```python
#!/usr/bin/env python3
import asyncio
import time
from gpu_worker_pool.client import GPUWorkerPoolClient

async def monitor_system():
    client = GPUWorkerPoolClient()
    
    try:
        await client.start()
        
        while True:
            print(f"\n--- System Status at {time.strftime('%Y-%m-%d %H:%M:%S')} ---")
            
            # Pool status
            status = client.get_pool_status()
            print(f"GPUs: {status.available_gpus}/{status.total_gpus} available")
            print(f"Workers: {status.active_workers} active, {status.blocked_workers} blocked")
            
            # Endpoint health (multi-endpoint mode)
            if client.is_multi_endpoint_mode():
                recovery_status = client.get_error_recovery_status()
                health = recovery_status['endpoint_health_summary']
                print(f"Endpoints: {health['healthy_count']}/{health['total_endpoints']} healthy")
            
            # Detailed metrics
            metrics = client.get_detailed_metrics()
            if 'load_balancer' in metrics:
                lb = metrics['load_balancer']
                print(f"Load Balancer: {lb['total_requests']} requests, {lb['success_rate']}% success")
            
            await asyncio.sleep(5)
            
    except KeyboardInterrupt:
        print("\nMonitoring stopped")
    finally:
        await client.stop()

asyncio.run(monitor_system())
```

### Common Error Messages

| Error | Meaning | Solution |
|-------|---------|----------|
| `RuntimeError: Client must be started` | Client not initialized | Call `await client.start()` or use context manager |
| `WorkerTimeoutError` | No GPU available within timeout | Increase timeout or check GPU availability |
| `ServiceUnavailableError` | Can't connect to GPU service | Check service is running and network connectivity |
| `ValueError: Invalid endpoint URL` | Malformed endpoint URL | Check URL format (http://host:port) |
| `RuntimeError: Not in multi-endpoint mode` | Trying to use multi-endpoint features | Use `service_endpoints` parameter |

## Getting Help

If issues persist:

1. **Check logs** with debug logging enabled
2. **Verify configuration** using validation scripts
3. **Test connectivity** to all endpoints
4. **Review examples** in the `examples/` directory
5. **Check the test suite** for usage patterns