# GPU Statistics Server

A Python FastAPI server that monitors GPU statistics including the number of installed GPUs, memory usage, and utilization percentage for each GPU. This server is part of the GPU Worker Pool system and provides the statistics endpoint that the worker pool client uses for intelligent GPU allocation.

## Features

- **GPU Detection**: Automatically detects the number of installed NVIDIA GPUs
- **Memory Monitoring**: Reports total, used, and free memory for each GPU
- **Utilization Tracking**: Shows GPU utilization percentage
- **Temperature Monitoring**: Reports GPU temperature (if supported)
- **Power Monitoring**: Shows power draw and limits (if supported)
- **REST API**: Easy-to-use HTTP endpoints for integration
- **Interactive Documentation**: Swagger UI and ReDoc documentation
- **Async Support**: Built with FastAPI for high performance
- **Caching**: Configurable refresh interval to reduce nvidia-smi calls
- **Error Handling**: Graceful handling of missing nvidia-smi or unsupported features
- **CORS Support**: Configurable Cross-Origin Resource Sharing
- **Environment Configuration**: Full configuration via environment variables

## Requirements

- Python 3.8+
- NVIDIA GPU(s) with NVIDIA drivers installed
- `nvidia-smi` command-line tool (comes with NVIDIA drivers)
- FastAPI and dependencies (see requirements below)

## Dependencies

The server requires the following Python packages:

```bash
pip install fastapi uvicorn python-dotenv pydantic
```

Or if you have the GPU Worker Pool package installed:

```bash
pip install gpu-worker-pool
```

## Installation

### Option 1: Using the GPU Worker Pool Package

If you have installed the `gpu-worker-pool` package, the server is included:

```python
from gpu_worker_pool.gpu_server import app
import uvicorn

# Run the server
uvicorn.run(app, host="0.0.0.0", port=8000)
```

### Option 2: Direct Installation

1. **Ensure you have the required dependencies**:
   ```bash
   pip install fastapi uvicorn python-dotenv pydantic
   ```

2. **Configure environment variables** (optional):
   ```bash
   # Create a .env file with your configuration
   echo "HOST=0.0.0.0" > .env
   echo "PORT=8000" >> .env
   echo "LOG_LEVEL=info" >> .env
   ```

## Usage

### Starting the Server

#### Method 1: Direct Python execution (recommended)
```bash
python -m gpu_worker_pool.gpu_server
```

#### Method 2: Using Python import
```python
from gpu_worker_pool.gpu_server import app
import uvicorn

# Run with default settings
uvicorn.run(app, host="0.0.0.0", port=8000)

# Or run with custom settings
uvicorn.run(
    app,
    host="127.0.0.1",
    port=8080,
    reload=True,
    log_level="debug"
)
```

#### Method 3: Using Uvicorn command line
```bash
uvicorn gpu_worker_pool.gpu_server:app --host 0.0.0.0 --port 8000 --reload
```

#### Method 4: Standalone script
If you have the gpu_server.py file directly:
```bash
python gpu_server.py
```

Once the server is running, you can:
- **Access the API**: http://localhost:8000 (or your configured port)
- **View Swagger docs**: http://localhost:8000/docs  
- **View ReDoc docs**: http://localhost:8000/redoc
- **Check health**: http://localhost:8000/health

### Integration with GPU Worker Pool

The GPU Statistics Server is designed to work with the GPU Worker Pool client. The client connects to the `/gpu/summary` endpoint to get GPU statistics for intelligent allocation:

```python
import asyncio
from gpu_worker_pool import GPUWorkerPoolClient

async def main():
    # The client will connect to the GPU server at the configured endpoint
    async with GPUWorkerPoolClient(
        service_endpoint="http://localhost:8000"  # Your GPU server
    ) as client:
        assignment = await client.request_gpu()
        print(f"Assigned GPU {assignment.gpu_id}")
        await client.release_gpu(assignment)

asyncio.run(main())
```

### API Endpoints

The server provides both interactive documentation and REST endpoints:

- **Swagger UI**: `http://localhost:8000/docs` - Interactive API documentation
- **ReDoc**: `http://localhost:8000/redoc` - Alternative API documentation

#### 1. Home (`/`)
Returns information about available endpoints and server version.

**Example**:
```bash
curl http://localhost:8000/
```

**Response**:
```json
{
  "message": "GPU Statistics Server",
  "version": "1.0.0",
  "endpoints": {
    "/": "This help message",
    "/gpu/stats": "Get detailed GPU statistics",
    "/gpu/count": "Get number of installed GPUs", 
    "/gpu/summary": "Get summary of GPU usage",
    "/health": "Health check endpoint",
    "/config": "Get server configuration",
    "/docs": "Swagger API documentation",
    "/redoc": "ReDoc API documentation"
  }
}
```

#### 2. Configuration (`/config`)
Returns the current server configuration settings.

**Example**:
```bash
curl http://localhost:8000/config
```

**Response**:
```json
{
  "server": {
    "host": "0.0.0.0",
    "port": 8000,
    "reload": true,
    "log_level": "info",
    "access_log": true,
    "workers": 1
  },
  "api": {
    "title": "GPU Statistics Server",
    "version": "1.0.0",
    "docs_url": "/docs",
    "redoc_url": "/redoc",
    "enable_cors": true,
    "api_prefix": ""
  },
  "monitoring": {
    "refresh_interval": 1.0,
    "max_history": 100
  }
}
```

#### 3. GPU Statistics (`/gpu/stats`)
Returns detailed statistics for all GPUs.

**Example**:
```bash
curl http://localhost:8000/gpu/stats
```

**Response**:
```json
{
  "gpu_count": 2,
  "gpus": [
    {
      "gpu_id": 0,
      "name": "NVIDIA GeForce RTX 3080",
      "memory": {
        "total_mb": 10240,
        "used_mb": 2048,
        "free_mb": 8192,
        "usage_percent": 20.0
      },
      "utilization_percent": 15,
      "temperature_c": 45,
      "power": {
        "draw_w": 120.5,
        "limit_w": 320.0
      }
    }
  ],
  "timestamp": "2025-07-23T10:30:00.123456"
}
```

#### 4. GPU Count (`/gpu/count`)
Returns only the number of installed GPUs.

**Example**:
```bash
curl http://localhost:8000/gpu/count
```

**Response**:
```json
{
  "gpu_count": 2,
  "timestamp": "2025-07-23T10:30:00.123456"
}
```

#### 5. GPU Summary (`/gpu/summary`)
Returns a summary of GPU usage across all GPUs.

**Example**:
```bash
curl http://localhost:8000/gpu/summary
```

**Response**:
```json
{
  "gpu_count": 2,
  "total_memory_mb": 20480,
  "total_used_memory_mb": 4096,
  "total_memory_usage_percent": 20.0,
  "average_utilization_percent": 17.5,
  "gpus_summary": [
    {
      "gpu_id": 0,
      "name": "NVIDIA GeForce RTX 3080",
      "memory_usage_percent": 20.0,
      "utilization_percent": 15
    },
    {
      "gpu_id": 1,
      "name": "NVIDIA GeForce RTX 3080",
      "memory_usage_percent": 20.0,
      "utilization_percent": 20
    }
  ],
  "timestamp": "2025-07-23T10:30:00.123456"
}
```

#### 6. Health Check (`/health`)
Returns server health status and nvidia-smi availability.

**Example**:
```bash
curl http://localhost:8000/health
```

**Response**:
```json
{
  "status": "healthy",
  "nvidia_smi_available": true,
  "timestamp": "2024-01-15T10:30:00.123456"
}
```

## Configuration

The server can be configured using environment variables. The server automatically loads configuration from a `.env` file if present in the working directory.

### GPU Filtering with CUDA_VISIBLE_DEVICES

The server supports filtering GPU statistics based on the `CUDA_VISIBLE_DEVICES` environment variable, which is the standard NVIDIA CUDA mechanism for controlling GPU visibility. This feature is particularly useful in containerized environments, multi-tenant systems, or when you want to restrict GPU access to a specific subset of devices.

#### Supported Formats

The server supports multiple formats for specifying visible GPUs:

- **Comma-separated**: `CUDA_VISIBLE_DEVICES="0,1,2"`
- **Space-separated**: `CUDA_VISIBLE_DEVICES="0 1 2"`
- **Mixed separators**: `CUDA_VISIBLE_DEVICES="0,1 2"`
- **Single GPU**: `CUDA_VISIBLE_DEVICES="1"`
- **No GPUs**: `CUDA_VISIBLE_DEVICES=""` (empty string)
- **All GPUs**: `CUDA_VISIBLE_DEVICES` not set or `None`

#### Examples

**Show only GPUs 0 and 2:**
```bash
export CUDA_VISIBLE_DEVICES="0,2"
python -m gpu_worker_pool.gpu_server
```

**Show only GPU 1:**
```bash
export CUDA_VISIBLE_DEVICES="1"
python -m gpu_worker_pool.gpu_server
```

**Hide all GPUs (useful for testing):**
```bash
export CUDA_VISIBLE_DEVICES=""
python -m gpu_worker_pool.gpu_server
```

**Docker container example:**
```bash
docker run -e CUDA_VISIBLE_DEVICES="0,1" your-gpu-server-image
```

#### Behavior

When GPU filtering is active:
- All API endpoints (`/gpu/stats`, `/gpu/count`, `/gpu/summary`) return only filtered GPUs
- GPU IDs in responses maintain their original system GPU indices
- Memory and utilization calculations are based only on visible GPUs
- The `/config` endpoint shows current filtering configuration
- Invalid GPU indices are logged as warnings and skipped
- If all specified GPUs are invalid, the server falls back to showing all GPUs

#### Filtered vs Unfiltered Response Examples

**System with 4 GPUs, CUDA_VISIBLE_DEVICES="0,2"**

Unfiltered response (CUDA_VISIBLE_DEVICES not set):
```json
{
  "gpu_count": 4,
  "gpus": [
    {"gpu_id": 0, "name": "NVIDIA RTX 3080", "memory": {"usage_percent": 25.0}, "utilization_percent": 30},
    {"gpu_id": 1, "name": "NVIDIA RTX 3080", "memory": {"usage_percent": 50.0}, "utilization_percent": 60},
    {"gpu_id": 2, "name": "NVIDIA RTX 3080", "memory": {"usage_percent": 10.0}, "utilization_percent": 15},
    {"gpu_id": 3, "name": "NVIDIA RTX 3080", "memory": {"usage_percent": 80.0}, "utilization_percent": 90}
  ]
}
```

Filtered response (CUDA_VISIBLE_DEVICES="0,2"):
```json
{
  "gpu_count": 2,
  "gpus": [
    {"gpu_id": 0, "name": "NVIDIA RTX 3080", "memory": {"usage_percent": 25.0}, "utilization_percent": 30},
    {"gpu_id": 2, "name": "NVIDIA RTX 3080", "memory": {"usage_percent": 10.0}, "utilization_percent": 15}
  ]
}
```

**Configuration endpoint with filtering active:**
```json
{
  "server": {"host": "0.0.0.0", "port": 8000},
  "gpu_filtering": {
    "cuda_visible_devices": "0,2",
    "visible_gpu_ids": [0, 2],
    "filtering_active": true,
    "total_system_gpus": 4,
    "visible_gpu_count": 2
  }
}
```

### Available Environment Variables:

#### Server Configuration
- `HOST` - Server bind address (default: "0.0.0.0")
- `PORT` - Server port (default: 8000)
- `RELOAD` - Enable auto-reload for development (default: true)
- `LOG_LEVEL` - Logging level: debug, info, warning, error, critical (default: "info")
- `ACCESS_LOG` - Enable access logging (default: true)
- `WORKERS` - Number of worker processes (default: 1)

#### API Configuration
- `TITLE` - API title (default: "GPU Statistics Server")
- `DESCRIPTION` - API description (default: "A server that provides GPU statistics including memory usage and utilization")
- `VERSION` - API version (default: "1.0.0")
- `DOCS_URL` - Swagger UI path (default: "/docs")
- `REDOC_URL` - ReDoc path (default: "/redoc")
- `ENABLE_CORS` - Enable CORS (default: true)
- `API_PREFIX` - API prefix (default: "")

#### GPU Filtering Configuration
- `CUDA_VISIBLE_DEVICES` - Comma or space-separated list of GPU indices to make visible (default: not set, shows all GPUs)

#### Monitoring Configuration
- `REFRESH_INTERVAL` - GPU data cache interval in seconds (default: 1.0)
- `MAX_HISTORY` - Maximum historical readings (default: 100)

#### Logging Configuration
- `LOG_FORMAT` - Log message format (default: "%(asctime)s - %(name)s - %(levelname)s - %(message)s")

### Configuration Helper Functions

The server includes helper functions for parsing environment variables:

- `get_bool_env(key, default)` - Parse boolean values (true/1/yes/on)
- `get_float_env(key, default)` - Parse float values with fallback
- `get_int_env(key, default)` - Parse integer values with fallback

### Example .env file:
```env
# Server settings
HOST=0.0.0.0
PORT=8000
RELOAD=false
LOG_LEVEL=info
ACCESS_LOG=true
WORKERS=1

# API settings
TITLE=My GPU Statistics Server
VERSION=1.0.0
ENABLE_CORS=true

# Monitoring settings
REFRESH_INTERVAL=1.0
MAX_HISTORY=100

# Logging
LOG_FORMAT=%(asctime)s - %(name)s - %(levelname)s - %(message)s
```

### Production Configuration Example:
```env
# Production settings
HOST=127.0.0.1
PORT=8000
RELOAD=false
LOG_LEVEL=warning
ACCESS_LOG=false
WORKERS=4
ENABLE_CORS=false
REFRESH_INTERVAL=2.0
```

### Development Configuration Example:
```env
# Development settings
HOST=0.0.0.0
PORT=8000
RELOAD=true
LOG_LEVEL=debug
ACCESS_LOG=true
WORKERS=1
ENABLE_CORS=true
REFRESH_INTERVAL=0.5
```
REFRESH_INTERVAL = 0.5  # Query GPU every 500ms
```

## Error Handling

The server handles various error conditions gracefully:

- **No NVIDIA GPUs**: Returns appropriate error messages
- **nvidia-smi not available**: Gracefully handles missing nvidia-smi
- **Permission issues**: Reports if nvidia-smi cannot be executed
- **Invalid requests**: Returns proper HTTP error codes
- **Unsupported GPU metrics**: Handles `[N/A]` and `[Not Supported]` values from nvidia-smi
- **Data conversion errors**: Safely converts string values to numbers with fallback defaults

### Common nvidia-smi Values Handled

The server properly handles these common nvidia-smi output values:
- `[Not Supported]` - Feature not supported by the GPU
- `[N/A]` - Data not available at query time
- Empty values or malformed data
- Temperature, power, and memory values that may be unavailable

## Example Client Code

### Python Client Example

```python
import requests
import json

# Get GPU statistics
response = requests.get('http://localhost:8000/gpu/stats')
if response.status_code == 200:
    gpu_data = response.json()
    print(f"Found {gpu_data['gpu_count']} GPU(s)")
    
    for gpu in gpu_data['gpus']:
        print(f"GPU {gpu['gpu_id']}: {gpu['name']}")
        print(f"  Memory: {gpu['memory']['used_mb']}MB / {gpu['memory']['total_mb']}MB ({gpu['memory']['usage_percent']}%)")
        print(f"  Utilization: {gpu['utilization_percent']}%")
        if gpu['temperature_c'] is not None:
            print(f"  Temperature: {gpu['temperature_c']}°C")
        if gpu['power']['draw_w'] is not None:
            print(f"  Power: {gpu['power']['draw_w']}W / {gpu['power']['limit_w']}W")
else:
    print(f"Error: {response.status_code}")

# Check server health
health_response = requests.get('http://localhost:8000/health')
if health_response.status_code == 200:
    health_data = health_response.json()
    print(f"Server status: {health_data['status']}")
    print(f"nvidia-smi available: {health_data['nvidia_smi_available']}")
```

### JavaScript/Node.js Client Example

```javascript
const axios = require('axios');

async function getGPUStats() {
    try {
        const response = await axios.get('http://localhost:8000/gpu/summary');
        const data = response.data;
        
        console.log(`Found ${data.gpu_count} GPU(s)`);
        console.log(`Total Memory Usage: ${data.total_memory_usage_percent}%`);
        console.log(`Average Utilization: ${data.average_utilization_percent}%`);
        
        data.gpus_summary.forEach(gpu => {
            console.log(`GPU ${gpu.gpu_id}: ${gpu.name} - ${gpu.utilization_percent}% utilization`);
        });
    } catch (error) {
        console.error('Error fetching GPU stats:', error.message);
    }
}

async function checkHealth() {
    try {
        const response = await axios.get('http://localhost:8000/health');
        const data = response.data;
        console.log(`Server status: ${data.status}`);
        console.log(`nvidia-smi available: ${data.nvidia_smi_available}`);
    } catch (error) {
        console.error('Health check failed:', error.message);
    }
}

// Usage
getGPUStats();
checkHealth();
```

## Troubleshooting

### Common Issues

1. **"nvidia-smi not found"**
   - Ensure NVIDIA drivers are installed
   - Verify nvidia-smi is in your PATH: `which nvidia-smi`

2. **Permission denied**
   - Run with appropriate permissions
   - Check if nvidia-smi requires sudo access

3. **No GPUs detected**
   - Verify GPUs are properly installed and recognized by the system
   - Run `nvidia-smi` directly to test

4. **Connection refused**
   - Check if the server is running
   - Verify the correct host and port
   - Check firewall settings

### GPU Filtering Issues

5. **GPUs not being filtered as expected**
   - Check the CUDA_VISIBLE_DEVICES environment variable: `echo $CUDA_VISIBLE_DEVICES`
   - Verify the server logs for filtering configuration at startup
   - Check the `/config` endpoint to see current filtering settings
   - Ensure GPU indices in CUDA_VISIBLE_DEVICES exist on your system

6. **Invalid GPU indices warnings**
   - Check server logs for warnings like "GPU X in CUDA_VISIBLE_DEVICES not found on system"
   - Run `nvidia-smi` to see available GPU indices (0, 1, 2, etc.)
   - Update CUDA_VISIBLE_DEVICES to use only valid GPU indices

7. **Server shows all GPUs despite CUDA_VISIBLE_DEVICES being set**
   - Verify the environment variable is set in the same shell/process as the server
   - Check for parsing errors in server logs
   - Ensure CUDA_VISIBLE_DEVICES format is correct (comma or space-separated numbers)
   - Try setting LOG_LEVEL=debug for more detailed filtering logs

8. **Empty GPU list when CUDA_VISIBLE_DEVICES is set**
   - Check if CUDA_VISIBLE_DEVICES is set to an empty string: `CUDA_VISIBLE_DEVICES=""`
   - Verify all GPU indices in CUDA_VISIBLE_DEVICES are valid
   - Check server logs for parsing errors or warnings

9. **Inconsistent GPU counts across endpoints**
   - All endpoints should return the same filtered GPU count
   - Check the `/config` endpoint to verify filtering is active
   - Restart the server if filtering configuration has changed

**Debugging GPU Filtering:**

Enable debug logging to see detailed filtering information:
```bash
export LOG_LEVEL=debug
export CUDA_VISIBLE_DEVICES="0,2"
python -m gpu_worker_pool.gpu_server
```

Check filtering configuration via API:
```bash
curl http://localhost:8000/config | jq '.gpu_filtering'
```

Verify current environment variable:
```bash
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
```

## License

This project is open source and available under the MIT License.
