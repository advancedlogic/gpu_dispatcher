# Installation Guide

## Installing from Wheel

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Install from Local Wheel

If you have the wheel file locally:

```bash
pip install gpu_worker_pool-1.0.0-py3-none-any.whl
```

### Install from Source

If you want to install from source:

```bash
# Clone the repository
git clone <repository-url>
cd gpu-worker-pool

# Install in development mode
pip install -e .

# Or build and install
python -m build
pip install dist/gpu_worker_pool-1.0.0-py3-none-any.whl
```

### Verify Installation

Test that the installation was successful:

```python
import gpu_worker_pool
print(f"GPU Worker Pool v{gpu_worker_pool.__version__} installed successfully")
```

Or run the test script:

```bash
python test_package.py
```

### Command Line Interface

After installation, you can use the command-line interface:

```bash
gpu-worker-pool-status
```

## Dependencies

The package has the following dependencies:

### Core Dependencies
- **fastapi** (>=0.68.0): For the GPU statistics server
- **uvicorn** (>=0.15.0): ASGI server for running the GPU server
- **pydantic** (>=1.8.0): Data validation and serialization
- **python-dotenv**: Environment variable management
- **aiohttp** (>=3.8.0): For HTTP client functionality

### System Dependencies
- **nvidia-smi**: Required for GPU monitoring (comes with NVIDIA drivers)
- **NVIDIA GPU drivers**: Required for real GPU monitoring

## Optional Dependencies

For development:

```bash
pip install gpu-worker-pool[dev]
```

This includes:
- pytest and pytest-asyncio for testing
- black and isort for code formatting
- flake8 and mypy for linting and type checking
- pre-commit for git hooks

For documentation:

```bash
pip install gpu-worker-pool[docs]
```

This includes:
- sphinx for documentation generation
- sphinx-rtd-theme for documentation theme

For examples:

```bash
pip install gpu-worker-pool[examples]
```

This includes:
- jupyter for running example notebooks
- matplotlib and pandas for data visualization

## Configuration

### GPU Worker Pool Client Configuration

Configure the GPU Worker Pool client using environment variables:

```bash
# Required: GPU statistics service endpoint
export GPU_SERVICE_ENDPOINT="http://your-gpu-service:8000"

# Optional: Resource thresholds (defaults shown)
export GPU_MEMORY_THRESHOLD_PERCENT="80.0"
export GPU_UTILIZATION_THRESHOLD_PERCENT="90.0"
export GPU_POLLING_INTERVAL="5"
```

### GPU Statistics Server Configuration

Configure the GPU statistics server using environment variables:

```bash
# Server settings
export HOST=0.0.0.0
export PORT=8000
export RELOAD=true
export LOG_LEVEL=info
export ACCESS_LOG=true
export WORKERS=1

# API settings
export TITLE="GPU Statistics Server"
export VERSION=1.0.0
export ENABLE_CORS=true

# Monitoring settings
export REFRESH_INTERVAL=1.0
export MAX_HISTORY=100

# Logging
export LOG_FORMAT="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
```

## GPU Statistics Server Setup

### Starting the Server

After installation, you can start the GPU statistics server using several methods:

#### Method 1: Direct Python execution (recommended)
```bash
python -m gpu_worker_pool.gpu_server
```

#### Method 2: Using uvicorn
```bash
uvicorn gpu_worker_pool.gpu_server:app --host 0.0.0.0 --port 8000 --reload
```

#### Method 3: Standalone script
```bash
python gpu_worker_pool/gpu_server.py
```

### Verify Server Installation

Test that the server is running correctly:

```bash
# Check server health
curl http://localhost:8000/health

# Get GPU statistics
curl http://localhost:8000/gpu/summary

# View API documentation
# Open http://localhost:8000/docs in your browser
```

### Server Requirements

For the GPU statistics server to work properly:

1. **NVIDIA GPU drivers must be installed**
2. **nvidia-smi must be available in PATH**
3. **Proper permissions to access GPU information**

Test nvidia-smi availability:
```bash
nvidia-smi --version
```

If nvidia-smi is not available, the server will still run but will return error responses for GPU endpoints.

## Quick Start

### GPU Worker Pool Client

```python
import asyncio
from gpu_worker_pool import GPUWorkerPoolClient

async def main():
    async with GPUWorkerPoolClient() as client:
        assignment = await client.request_gpu()
        print(f"Got GPU {assignment.gpu_id}")
        
        # Do your work here
        await asyncio.sleep(1.0)
        
        await client.release_gpu(assignment)
        print("GPU released")

asyncio.run(main())
```

### GPU Statistics Server

```python
# Start the server programmatically
from gpu_worker_pool.gpu_server import app
import uvicorn

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### Complete Setup Example

1. **Start the GPU statistics server:**
```bash
# Terminal 1: Start the server
python -m gpu_worker_pool.gpu_server
```

2. **Use the client in another terminal:**
```python
# Terminal 2: Use the client
import asyncio
from gpu_worker_pool import GPUWorkerPoolClient

async def main():
    # Configure client to use local server
    async with GPUWorkerPoolClient(
        service_endpoint="http://localhost:8000"
    ) as client:
        assignment = await client.request_gpu()
        print(f"Got GPU {assignment.gpu_id}")
        await client.release_gpu(assignment)

asyncio.run(main())
```

## Troubleshooting

### Import Errors

If you get import errors, make sure you have Python 3.8+ and the package is properly installed:

```bash
python --version
pip show gpu-worker-pool
```

### Service Connection Issues

If you can't connect to the GPU service:

1. Check that the service endpoint is correct
2. Verify network connectivity
3. Check firewall settings
4. Ensure the GPU service is running

### Permission Issues

If you get permission errors during installation:

```bash
# Install for current user only
pip install --user gpu_worker_pool-1.0.0-py3-none-any.whl

# Or use virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install gpu_worker_pool-1.0.0-py3-none-any.whl
```

## Uninstallation

To remove the package:

```bash
pip uninstall gpu-worker-pool
```