#!/usr/bin/env python3
"""
GPU Statistics Server

A FastAPI server that provides GPU statistics including:
- Number of installed GPUs
- Memory usage for each GPU
- GPU utilization percentage for each GPU
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import logging
import subprocess
import json
import os
from typing import Dict, List, Any, Optional
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Helper function to parse boolean environment variables
def get_bool_env(key: str, default: bool = False) -> bool:
    value = os.getenv(key, str(default)).lower()
    return value in ('true', '1', 'yes', 'on')

# Helper function to parse float environment variables
def get_float_env(key: str, default: float) -> float:
    try:
        return float(os.getenv(key, str(default)))
    except ValueError:
        return default

# Helper function to parse int environment variables
def get_int_env(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, str(default)))
    except ValueError:
        return default

# Load configuration from environment variables
HOST = os.getenv("HOST", "0.0.0.0")
PORT = get_int_env("PORT", 8000)
RELOAD = get_bool_env("RELOAD", True)
LOG_LEVEL = os.getenv("LOG_LEVEL", "info")
ACCESS_LOG = get_bool_env("ACCESS_LOG", True)
WORKERS = get_int_env("WORKERS", 1)
TITLE = os.getenv("TITLE", "GPU Statistics Server")
DESCRIPTION = os.getenv("DESCRIPTION", "A server that provides GPU statistics including memory usage and utilization")
VERSION = os.getenv("VERSION", "1.0.0")
DOCS_URL = os.getenv("DOCS_URL", "/docs")
REDOC_URL = os.getenv("REDOC_URL", "/redoc")
REFRESH_INTERVAL = get_float_env("REFRESH_INTERVAL", 1.0)
MAX_HISTORY = get_int_env("MAX_HISTORY", 100)
ENABLE_CORS = get_bool_env("ENABLE_CORS", True)
API_PREFIX = os.getenv("API_PREFIX", "")
LOG_FORMAT = os.getenv("LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")

# Configure logging using environment values
logging.basicConfig(level=getattr(logging, LOG_LEVEL.upper()), format=LOG_FORMAT)
logger = logging.getLogger(__name__)

# Pydantic models for API documentation
class GPUMemory(BaseModel):
    total_mb: int
    used_mb: int
    free_mb: int
    usage_percent: float

class GPUPower(BaseModel):
    draw_w: Optional[float]
    limit_w: Optional[float]

class GPUInfo(BaseModel):
    gpu_id: int
    name: str
    memory: GPUMemory
    utilization_percent: int
    temperature_c: Optional[int]
    power: GPUPower

class GPUStats(BaseModel):
    gpu_count: int
    gpus: List[GPUInfo]
    timestamp: str

class GPUSummaryItem(BaseModel):
    gpu_id: int
    name: str
    memory_usage_percent: float
    utilization_percent: int

class GPUSummary(BaseModel):
    gpu_count: int
    total_memory_mb: int
    total_used_memory_mb: int
    total_memory_usage_percent: float
    average_utilization_percent: float
    gpus_summary: List[GPUSummaryItem]
    timestamp: str

class HealthCheck(BaseModel):
    status: str
    nvidia_smi_available: bool
    timestamp: str

# Create FastAPI app using environment variables
app = FastAPI(
    title=TITLE,
    description=DESCRIPTION,
    version=VERSION,
    docs_url=DOCS_URL,
    redoc_url=REDOC_URL
)

# Add CORS middleware if enabled
if ENABLE_CORS:
    from fastapi.middleware.cors import CORSMiddleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure this more restrictively in production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


class GPUMonitor:
    """Class to handle GPU monitoring operations"""
    
    def __init__(self):
        self.nvidia_smi_available = self._check_nvidia_smi()
        self._last_update = 0
        self._cached_stats = None
        self._refresh_interval = REFRESH_INTERVAL
    
    def _check_nvidia_smi(self) -> bool:
        """Check if nvidia-smi is available on the system"""
        try:
            subprocess.run(['nvidia-smi', '--version'], 
                         capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.warning("nvidia-smi not found. GPU monitoring may not work.")
            return False
    
    def get_gpu_stats(self) -> Dict[str, Any]:
        """
        Get GPU statistics using nvidia-smi with caching based on REFRESH_INTERVAL
        
        Returns:
            Dict containing GPU count and statistics for each GPU
        """
        import time
        current_time = time.time()
        
        # Use cached data if within refresh interval
        if (self._cached_stats is not None and 
            current_time - self._last_update < self._refresh_interval):
            return self._cached_stats
        
        # Update stats
        self._cached_stats = self._fetch_gpu_stats()
        self._last_update = current_time
        return self._cached_stats
    
    def _fetch_gpu_stats(self) -> Dict[str, Any]:
        """
        Fetch GPU statistics using nvidia-smi
        
        Returns:
            Dict containing GPU count and statistics for each GPU
        """
        if not self.nvidia_smi_available:
            return {
                "error": "nvidia-smi not available",
                "gpu_count": 0,
                "gpus": []
            }
        
        try:
            # Run nvidia-smi with JSON output
            cmd = [
                'nvidia-smi',
                '--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu,power.draw,power.limit',
                '--format=csv,noheader,nounits'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            gpus = []
            lines = result.stdout.strip().split('\n')
            
            for line in lines:
                if line.strip():
                    parts = [part.strip() for part in line.split(',')]
                    if len(parts) >= 6:
                        # Helper function to safely convert values
                        def safe_int(value, default=0):
                            if value in ['[Not Supported]', '[N/A]', 'N/A', '']:
                                return default
                            try:
                                return int(value)
                            except (ValueError, TypeError):
                                return default
                        
                        def safe_float(value, default=None):
                            if value in ['[Not Supported]', '[N/A]', 'N/A', '']:
                                return default
                            try:
                                return float(value)
                            except (ValueError, TypeError):
                                return default
                        
                        # Parse memory values safely
                        total_mb = safe_int(parts[2], 0)
                        used_mb = safe_int(parts[3], 0)
                        free_mb = safe_int(parts[4], 0)
                        
                        # Calculate memory usage percentage safely
                        usage_percent = 0
                        if total_mb > 0 and used_mb >= 0:
                            usage_percent = round((used_mb / total_mb) * 100, 2)
                        
                        gpu_info = {
                            "gpu_id": safe_int(parts[0], 0),
                            "name": parts[1] if parts[1] not in ['[Not Supported]', '[N/A]'] else "Unknown GPU",
                            "memory": {
                                "total_mb": total_mb,
                                "used_mb": used_mb,
                                "free_mb": free_mb,
                                "usage_percent": usage_percent
                            },
                            "utilization_percent": safe_int(parts[5], 0),
                            "temperature_c": safe_int(parts[6], None) if len(parts) > 6 else None,
                            "power": {
                                "draw_w": safe_float(parts[7], None) if len(parts) > 7 else None,
                                "limit_w": safe_float(parts[8], None) if len(parts) > 8 else None
                            }
                        }
                        gpus.append(gpu_info)
            
            return {
                "gpu_count": len(gpus),
                "gpus": gpus,
                "timestamp": self._get_timestamp()
            }
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Error running nvidia-smi: {e}")
            return {
                "error": f"Failed to get GPU stats: {e}",
                "gpu_count": 0,
                "gpus": []
            }
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return {
                "error": f"Unexpected error: {e}",
                "gpu_count": 0,
                "gpus": []
            }
    
    def _get_timestamp(self) -> str:
        """Get current timestamp as ISO format string"""
        from datetime import datetime
        return datetime.now().isoformat()


# Initialize GPU monitor
gpu_monitor = GPUMonitor()


@app.get("/", tags=["Info"])
async def home() -> Dict[str, Any]:
    """Home endpoint with API information"""
    return {
        "message": "GPU Statistics Server",
        "version": VERSION,
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


@app.get("/gpu/stats", tags=["GPU"], response_model=None)
async def gpu_stats() -> Dict[str, Any]:
    """Get detailed GPU statistics"""
    stats = gpu_monitor.get_gpu_stats()
    return stats


@app.get("/gpu/count", tags=["GPU"]) 
async def gpu_count() -> Dict[str, Any]:
    """Get number of installed GPUs"""
    stats = gpu_monitor.get_gpu_stats()
    return {
        "gpu_count": stats.get("gpu_count", 0),
        "timestamp": stats.get("timestamp")
    }


@app.get("/gpu/summary", tags=["GPU"], response_model=None)
async def gpu_summary() -> Dict[str, Any]:
    """Get summary of GPU usage"""
    stats = gpu_monitor.get_gpu_stats()
    
    if "error" in stats:
        return stats
    
    summary = {
        "gpu_count": stats["gpu_count"],
        "total_memory_mb": 0,
        "total_used_memory_mb": 0,
        "average_utilization_percent": 0,
        "gpus_summary": []
    }
    
    if stats["gpus"]:
        total_util = 0
        for gpu in stats["gpus"]:
            summary["total_memory_mb"] += gpu["memory"]["total_mb"]
            summary["total_used_memory_mb"] += gpu["memory"]["used_mb"]
            total_util += gpu["utilization_percent"]
            
            summary["gpus_summary"].append({
                "gpu_id": gpu["gpu_id"],
                "name": gpu["name"],
                "memory_usage_percent": gpu["memory"]["usage_percent"],
                "utilization_percent": gpu["utilization_percent"]
            })
        
        summary["average_utilization_percent"] = round(total_util / len(stats["gpus"]), 2)
        summary["total_memory_usage_percent"] = round(
            (summary["total_used_memory_mb"] / summary["total_memory_mb"]) * 100, 2
        ) if summary["total_memory_mb"] > 0 else 0
    
    summary["timestamp"] = stats.get("timestamp")
    return summary


@app.get("/health", tags=["Health"], response_model=None)
async def health() -> Dict[str, Any]:
    """Health check endpoint"""
    return {
        "status": "healthy",
        "nvidia_smi_available": gpu_monitor.nvidia_smi_available,
        "timestamp": gpu_monitor._get_timestamp()
    }


@app.get("/config", tags=["Info"])
async def get_config() -> Dict[str, Any]:
    """Get server configuration information"""
    return {
        "server": {
            "host": HOST,
            "port": PORT,
            "reload": RELOAD,
            "log_level": LOG_LEVEL,
            "access_log": ACCESS_LOG,
            "workers": WORKERS
        },
        "api": {
            "title": TITLE,
            "version": VERSION,
            "docs_url": DOCS_URL,
            "redoc_url": REDOC_URL,
            "enable_cors": ENABLE_CORS,
            "api_prefix": API_PREFIX
        },
        "monitoring": {
            "refresh_interval": REFRESH_INTERVAL,
            "max_history": MAX_HISTORY
        }
    }


if __name__ == '__main__':
    import uvicorn
    
    logger.info("Starting GPU Statistics Server...")
    logger.info(f"nvidia-smi available: {gpu_monitor.nvidia_smi_available}")
    logger.info(f"Server configuration (from .env file):")
    logger.info(f"  Host: {HOST}")
    logger.info(f"  Port: {PORT}")
    logger.info(f"  Reload: {RELOAD}")
    logger.info(f"  Log Level: {LOG_LEVEL}")
    logger.info(f"  Refresh Interval: {REFRESH_INTERVAL}s")
    
    # Run the FastAPI server with Uvicorn using environment variables
    uvicorn.run(
        "gpu_server:app",
        host=HOST,
        port=PORT,
        reload=RELOAD,
        log_level=LOG_LEVEL.lower(),
        access_log=ACCESS_LOG,
        workers=1 if RELOAD else WORKERS  # Use 1 worker in reload mode
    )
