# Migration Guide: Single-Endpoint to Multi-Endpoint

This guide helps existing users migrate from single-endpoint to multi-endpoint configurations in the GPU Worker Pool.

## Table of Contents

- [Overview](#overview)
- [Why Migrate?](#why-migrate)
- [Migration Steps](#migration-steps)
- [Configuration Changes](#configuration-changes)
- [Code Changes](#code-changes)
- [Backward Compatibility](#backward-compatibility)
- [Common Scenarios](#common-scenarios)
- [Troubleshooting](#troubleshooting)

## Overview

The multi-endpoint feature allows the GPU Worker Pool to connect to multiple GPU statistics servers simultaneously, providing:

- **Increased Capacity**: Access GPUs from multiple servers
- **High Availability**: Automatic failover when servers become unavailable
- **Load Balancing**: Intelligent distribution of GPU requests
- **Better Resource Utilization**: Choose GPUs from the server with most availability

## Why Migrate?

Consider migrating to multi-endpoint if you:

1. **Need More GPUs**: Your workload exceeds a single server's capacity
2. **Want High Availability**: Eliminate single points of failure
3. **Have Distributed Infrastructure**: GPUs spread across multiple servers
4. **Need Better Performance**: Reduce contention on individual servers

## Migration Steps

### Step 1: Assess Your Current Setup

First, identify your current configuration:

```python
# Current single-endpoint setup
client = GPUWorkerPoolClient(
    service_endpoint="http://gpu-server:8000",
    memory_threshold=75.0,
    utilization_threshold=85.0
)
```

Or via environment variables:
```bash
export GPU_SERVICE_ENDPOINT="http://gpu-server:8000"
```

### Step 2: Plan Your Multi-Endpoint Architecture

Decide on:
- How many GPU servers you'll use
- Which load balancing strategy fits your needs
- Network connectivity between clients and servers

### Step 3: Update Your Configuration

#### Option A: Code-Based Configuration

**Before (Single-Endpoint):**
```python
client = GPUWorkerPoolClient(
    service_endpoint="http://gpu-server:8000",
    memory_threshold=75.0,
    utilization_threshold=85.0
)
```

**After (Multi-Endpoint):**
```python
client = GPUWorkerPoolClient(
    service_endpoints="http://gpu-server-1:8000,http://gpu-server-2:8000,http://gpu-server-3:8000",
    load_balancing_strategy="availability",  # Optional, this is the default
    memory_threshold=75.0,
    utilization_threshold=85.0
)
```

#### Option B: Environment Variable Configuration

**Before (Single-Endpoint):**
```bash
export GPU_SERVICE_ENDPOINT="http://gpu-server:8000"
export GPU_MEMORY_THRESHOLD_PERCENT="75.0"
export GPU_UTILIZATION_THRESHOLD_PERCENT="85.0"
```

**After (Multi-Endpoint):**
```bash
export GPU_STATS_SERVICE_ENDPOINTS="http://gpu-server-1:8000,http://gpu-server-2:8000,http://gpu-server-3:8000"
export GPU_LOAD_BALANCING_STRATEGY="availability"
export GPU_MEMORY_THRESHOLD_PERCENT="75.0"
export GPU_UTILIZATION_THRESHOLD_PERCENT="85.0"
```

### Step 4: Update Your Code (If Needed)

Most code works without changes, but be aware of these differences:

#### GPU IDs

**Single-Endpoint:** GPU IDs are simple integers (0, 1, 2, ...)

**Multi-Endpoint:** GPU IDs are global identifiers ("server1:0", "server1:1", "server2:0", ...)

```python
# This works in both modes
assignment = await client.request_gpu()
print(f"Assigned GPU: {assignment.gpu_id}")  # "0" or "server1:0"

# Release works the same way
await client.release_gpu(assignment)
```

#### Pool Status

In multi-endpoint mode, `get_pool_status()` returns `MultiEndpointPoolStatus` with additional fields:

```python
status = client.get_pool_status()

# These work in both modes
print(f"Total GPUs: {status.total_gpus}")
print(f"Available GPUs: {status.available_gpus}")

# These are only available in multi-endpoint mode
if hasattr(status, 'total_endpoints'):
    print(f"Total endpoints: {status.total_endpoints}")
    print(f"Healthy endpoints: {status.healthy_endpoints}")
```

### Step 5: Test Your Migration

1. **Start with one endpoint** to ensure compatibility:
   ```python
   # This uses multi-endpoint mode with a single endpoint
   client = GPUWorkerPoolClient(
       service_endpoints="http://gpu-server:8000"
   )
   ```

2. **Add endpoints gradually**:
   ```python
   # Add a second endpoint
   client = GPUWorkerPoolClient(
       service_endpoints="http://gpu-server-1:8000,http://gpu-server-2:8000"
   )
   ```

3. **Monitor the system**:
   ```python
   # Check endpoint health
   endpoints = client.get_endpoints_info()
   for endpoint in endpoints:
       print(f"{endpoint['endpoint_id']}: {endpoint['is_healthy']}")
   ```

## Configuration Changes

### Environment Variable Mapping

| Single-Endpoint | Multi-Endpoint | Notes |
|----------------|----------------|-------|
| `GPU_SERVICE_ENDPOINT` | `GPU_STATS_SERVICE_ENDPOINTS` | Use comma-separated URLs |
| N/A | `GPU_LOAD_BALANCING_STRATEGY` | New: Choose strategy |
| Same | Same | All other variables unchanged |

### Configuration Priority

When both single and multi-endpoint variables are set:
1. `GPU_STATS_SERVICE_ENDPOINTS` takes precedence
2. `service_endpoints` parameter takes precedence over `service_endpoint`

## Code Changes

### Minimal Changes Required

Most existing code works without modification:

```python
# This code works in both single and multi-endpoint modes
async with GPUWorkerPoolClient() as client:
    assignment = await client.request_gpu()
    # Do work...
    await client.release_gpu(assignment)
```

### Optional Enhancements

Take advantage of multi-endpoint features:

```python
# Check if in multi-endpoint mode
if client.is_multi_endpoint_mode():
    # Get endpoint information
    endpoints = client.get_endpoints_info()
    
    # Monitor health
    recovery_status = client.get_error_recovery_status()
    
    # Trigger manual recovery
    await client.trigger_endpoint_recovery("server2")
```

## Backward Compatibility

### Full Compatibility Mode

To ensure full backward compatibility:

1. **Keep using `service_endpoint`** (singular) for single-endpoint mode
2. **Use `service_endpoints`** (plural) only when you need multi-endpoint features
3. **All existing code continues to work** without changes

### API Compatibility

| Feature | Single-Endpoint | Multi-Endpoint |
|---------|----------------|----------------|
| `request_gpu()` | Returns local GPU ID | Returns global GPU ID |
| `release_gpu()` | Works with local ID | Works with global ID |
| `get_pool_status()` | Returns `PoolStatus` | Returns `MultiEndpointPoolStatus` |
| Context managers | Fully supported | Fully supported |
| Error handling | Same exceptions | Same exceptions |

## Common Scenarios

### Scenario 1: Adding Redundancy

**Goal**: Add a backup GPU server for high availability

```python
# Primary + backup configuration
client = GPUWorkerPoolClient(
    service_endpoints="http://primary-gpu:8000,http://backup-gpu:8000",
    load_balancing_strategy="availability"
)
```

### Scenario 2: Scaling Horizontally

**Goal**: Distribute load across multiple GPU servers

```python
# Scale across multiple servers
endpoints = ",".join([f"http://gpu-node-{i}:8000" for i in range(1, 6)])
client = GPUWorkerPoolClient(
    service_endpoints=endpoints,
    load_balancing_strategy="round_robin"  # Even distribution
)
```

### Scenario 3: Heterogeneous Clusters

**Goal**: Different GPU types on different servers

```python
# Weighted distribution based on server capacity
client = GPUWorkerPoolClient(
    service_endpoints="http://v100-server:8000,http://a100-server:8000,http://t4-server:8000",
    load_balancing_strategy="weighted"  # Considers GPU count per server
)
```

### Scenario 4: Geographic Distribution

**Goal**: GPU servers in different locations

```python
# Multi-region setup
client = GPUWorkerPoolClient(
    service_endpoints="http://us-east-gpu:8000,http://us-west-gpu:8000,http://eu-gpu:8000",
    load_balancing_strategy="availability"
)
```

## Troubleshooting

### Issue: Client Not Detecting Multi-Endpoint Mode

**Symptom**: `client.is_multi_endpoint_mode()` returns `False`

**Solutions**:
1. Use `service_endpoints` (plural), not `service_endpoint`
2. Check environment variable name: `GPU_STATS_SERVICE_ENDPOINTS`
3. Verify comma-separated format: `"http://server1:8000,http://server2:8000"`

### Issue: Global GPU IDs Breaking Existing Code

**Symptom**: Code expects integer GPU IDs but gets strings like "server1:0"

**Solutions**:
1. Update logging/monitoring to handle string IDs
2. Use the ID as-is (the client handles routing automatically)
3. Extract local ID if needed:
   ```python
   global_id = "server1:2"
   local_id = int(global_id.split(':')[1])  # Gets 2
   ```

### Issue: Endpoints Not Connecting

**Symptom**: Some endpoints show as unhealthy

**Check**:
1. Network connectivity to each endpoint
2. GPU statistics server is running on each endpoint
3. Firewall rules allow connections
4. Use monitoring to check endpoint status:
   ```python
   client.print_error_recovery_summary()
   ```

### Issue: Uneven Load Distribution

**Symptom**: One server gets all requests while others are idle

**Solutions**:
1. Check load balancing strategy:
   - `availability`: Prefers servers with more free GPUs
   - `round_robin`: Even distribution
   - `weighted`: Based on total GPU count
2. Verify all endpoints are healthy
3. Check GPU availability on each server

### Issue: Performance Degradation

**Symptom**: Slower response times with multiple endpoints

**Solutions**:
1. Check network latency between client and servers
2. Adjust timeouts:
   ```bash
   export GPU_ENDPOINT_TIMEOUT="15.0"  # Increase for slow networks
   ```
3. Monitor endpoint response times:
   ```python
   endpoints = client.get_endpoints_info()
   for ep in endpoints:
       print(f"{ep['endpoint_id']}: {ep['response_time_ms']}ms")
   ```

## Best Practices

1. **Start Simple**: Begin with 2-3 endpoints and scale up
2. **Monitor Health**: Regularly check endpoint and system health
3. **Choose the Right Strategy**: 
   - `availability` for optimal utilization
   - `round_robin` for testing and even distribution
   - `weighted` for heterogeneous clusters
4. **Set Appropriate Timeouts**: Based on your network characteristics
5. **Test Failover**: Verify system behavior when endpoints fail
6. **Use Circuit Breakers**: They prevent cascading failures automatically

## Migration Checklist

- [ ] Identify current configuration (endpoint URL, thresholds)
- [ ] Choose target endpoints for multi-endpoint setup
- [ ] Select appropriate load balancing strategy
- [ ] Update configuration (code or environment variables)
- [ ] Test with single endpoint in multi-endpoint mode
- [ ] Add additional endpoints gradually
- [ ] Verify GPU assignments work correctly
- [ ] Check monitoring and metrics
- [ ] Test failover scenarios
- [ ] Update any ID-dependent code for global IDs
- [ ] Document your multi-endpoint configuration

## Support

For additional help:
- See [API Documentation](API.md) for detailed method descriptions
- Check [examples/multi_endpoint_usage.py](../examples/multi_endpoint_usage.py) for code examples
- Review the [README](../README.md) for configuration options