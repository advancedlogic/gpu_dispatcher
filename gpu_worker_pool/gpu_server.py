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
import re
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
VISIBLE_DEVICES = os.getenv("VISIBLE_DEVICES", "0")

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
        self._visible_devices = self._parse_visible_devices()
    
    def _check_nvidia_smi(self) -> bool:
        """Check if nvidia-smi is available on the system"""
        try:
            subprocess.run(['nvidia-smi', '--version'], 
                         capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.warning("nvidia-smi not found. GPU monitoring may not work.")
            return False
    
    def _parse_visible_devices(self) -> Optional[List[int]]:
        """
        Parse CUDA_VISIBLE_DEVICES environment variable to determine which GPUs should be visible.
        
        Returns:
            List of GPU indices that should be visible, or None if all GPUs should be visible
        """
        cuda_visible = os.getenv("CUDA_VISIBLE_DEVICES")
        
        # Log the CUDA_VISIBLE_DEVICES value for debugging
        logger.info(f"CUDA_VISIBLE_DEVICES environment variable: {repr(cuda_visible)}")
        
        # If not set, show all GPUs (current behavior)
        if cuda_visible is None:
            logger.info("CUDA_VISIBLE_DEVICES not set, showing all GPUs")
            return None
        
        # If empty string, show no GPUs
        if cuda_visible.strip() == "":
            logger.info("CUDA_VISIBLE_DEVICES is empty, showing no GPUs")
            return []
        
        try:
            devices = []
            invalid_parts = []
            negative_parts = []
            
            # Support comma-separated, space-separated, and mixed separators
            # Split on both commas and whitespace, then filter out empty strings
            parts = re.split(r'[,\s]+', cuda_visible.strip())
            
            for part in parts:
                part = part.strip()
                if not part:  # Skip empty parts
                    continue
                    
                try:
                    device_id = int(part)
                    if device_id < 0:
                        negative_parts.append(part)
                        logger.warning(f"Invalid GPU index '{part}' in CUDA_VISIBLE_DEVICES: negative GPU indices are not allowed")
                        continue
                    devices.append(device_id)
                except ValueError:
                    invalid_parts.append(part)
                    logger.warning(f"Invalid GPU index '{part}' in CUDA_VISIBLE_DEVICES: must be a non-negative integer")
                    continue
            
            # Log summary of parsing issues if any occurred
            if invalid_parts or negative_parts:
                all_invalid = invalid_parts + negative_parts
                logger.warning(f"CUDA_VISIBLE_DEVICES parsing found {len(all_invalid)} invalid entries: {all_invalid}")
                if devices:
                    logger.info(f"Continuing with {len(devices)} valid GPU indices: {devices}")
                else:
                    logger.error("CUDA_VISIBLE_DEVICES parsing failed: no valid GPU indices found")
                    logger.info("Falling back to showing all GPUs due to parsing errors")
                    return None
            
            if devices:
                logger.info(f"Successfully parsed {len(devices)} visible GPU IDs from CUDA_VISIBLE_DEVICES: {devices}")
                return devices
            else:
                logger.error("CUDA_VISIBLE_DEVICES parsing failed: no valid GPU IDs found after processing")
                logger.info("Falling back to showing all GPUs")
                return None
                
        except Exception as e:
            logger.error(f"Critical error parsing CUDA_VISIBLE_DEVICES '{cuda_visible}': {type(e).__name__}: {e}")
            logger.error("This indicates a serious parsing failure - falling back to showing all GPUs")
            return None
    
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
    
    def _filter_gpu_data(self, gpu_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Filter GPU data based on visible devices from CUDA_VISIBLE_DEVICES.
        
        Args:
            gpu_data: Raw GPU data from nvidia-smi
            
        Returns:
            Filtered GPU data containing only visible GPUs
        """
        # If no filtering is configured, return original data
        if self._visible_devices is None:
            return gpu_data
        
        # If there's an error in the original data, pass it through
        if "error" in gpu_data:
            return gpu_data
        
        # Get available GPU IDs from the raw data
        available_gpu_ids = {gpu["gpu_id"] for gpu in gpu_data.get("gpus", [])}
        valid_visible_devices = []
        missing_gpu_ids = []
        
        # Check each visible device and warn about non-existent ones
        for device_id in self._visible_devices:
            if device_id in available_gpu_ids:
                valid_visible_devices.append(device_id)
            else:
                missing_gpu_ids.append(device_id)
        
        # Log detailed warnings for missing GPUs
        if missing_gpu_ids:
            if len(missing_gpu_ids) == 1:
                logger.warning(f"GPU {missing_gpu_ids[0]} specified in CUDA_VISIBLE_DEVICES does not exist on this system")
            else:
                logger.warning(f"GPUs {missing_gpu_ids} specified in CUDA_VISIBLE_DEVICES do not exist on this system")
            
            logger.warning(f"Available GPU IDs on system: {sorted(available_gpu_ids) if available_gpu_ids else 'none'}")
            
            if not valid_visible_devices:
                logger.error("No valid GPUs found after filtering - all specified GPUs are missing from system")
                logger.info("Returning empty GPU list due to complete filtering mismatch")
            else:
                logger.info(f"Continuing with {len(valid_visible_devices)} valid GPUs: {valid_visible_devices}")
        
        # Filter GPUs to only include visible ones, maintaining original GPU IDs
        filtered_gpus = [
            gpu for gpu in gpu_data.get("gpus", [])
            if gpu["gpu_id"] in valid_visible_devices
        ]
        
        # Create filtered result with updated gpu_count but preserved metadata
        filtered_data = {
            "gpu_count": len(filtered_gpus),
            "gpus": filtered_gpus,
            "timestamp": gpu_data.get("timestamp")
        }
        
        # Log filtering results for debugging
        total_gpus = len(gpu_data.get("gpus", []))
        visible_gpus = len(filtered_gpus)
        
        if visible_gpus != total_gpus:
            logger.info(f"GPU filtering applied: {total_gpus} total system GPUs -> {visible_gpus} visible GPUs")
            if visible_gpus > 0:
                visible_ids = [gpu["gpu_id"] for gpu in filtered_gpus]
                logger.debug(f"Filtered GPU IDs: {visible_ids}")
        
        return filtered_data
    
    def _fetch_gpu_stats(self) -> Dict[str, Any]:
        """
        Fetch GPU statistics using nvidia-smi and apply filtering
        
        Returns:
            Dict containing GPU count and statistics for each GPU (filtered if CUDA_VISIBLE_DEVICES is set)
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
            
            # Create raw GPU data
            raw_gpu_data = {
                "gpu_count": len(gpus),
                "gpus": gpus,
                "timestamp": self._get_timestamp()
            }
            
            # Apply filtering and return filtered data
            return self._filter_gpu_data(raw_gpu_data)
            
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
    # Get current GPU stats to determine total system GPUs
    stats = gpu_monitor.get_gpu_stats()
    
    # Calculate total system GPUs (this would be the unfiltered count)
    # We need to temporarily get unfiltered data to show total system GPUs
    total_system_gpus = 0
    if gpu_monitor.nvidia_smi_available:
        try:
            # Get unfiltered GPU count by temporarily bypassing filtering
            original_visible_devices = gpu_monitor._visible_devices
            gpu_monitor._visible_devices = None  # Temporarily disable filtering
            unfiltered_stats = gpu_monitor._fetch_gpu_stats()
            total_system_gpus = unfiltered_stats.get("gpu_count", 0)
            gpu_monitor._visible_devices = original_visible_devices  # Restore filtering
        except Exception:
            total_system_gpus = stats.get("gpu_count", 0)
    
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
        },
        "gpu_filtering": {
            "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
            "visible_gpu_ids": gpu_monitor._visible_devices,
            "filtering_active": gpu_monitor._visible_devices is not None,
            "total_system_gpus": total_system_gpus,
            "visible_gpu_count": stats.get("gpu_count", 0)
        }
    }


if __name__ == '__main__':
    import uvicorn
    
    logger.info("=" * 60)
    logger.info("Starting GPU Statistics Server...")
    logger.info("=" * 60)
    
    # Log nvidia-smi availability
    logger.info(f"nvidia-smi available: {gpu_monitor.nvidia_smi_available}")
    
    # Enhanced startup logging to show CUDA_VISIBLE_DEVICES configuration
    cuda_visible_devices = os.getenv("CUDA_VISIBLE_DEVICES")
    logger.info(f"CUDA_VISIBLE_DEVICES environment variable: {repr(cuda_visible_devices)}")
    
    # Get system GPU information for comprehensive logging
    total_system_gpus = 0
    if gpu_monitor.nvidia_smi_available:
        try:
            # Temporarily disable filtering to get total system GPU count
            original_visible_devices = gpu_monitor._visible_devices
            gpu_monitor._visible_devices = None
            unfiltered_stats = gpu_monitor._fetch_gpu_stats()
            total_system_gpus = unfiltered_stats.get("gpu_count", 0)
            gpu_monitor._visible_devices = original_visible_devices
        except Exception as e:
            logger.warning(f"Could not determine total system GPU count: {e}")
    
    # Log filtering configuration details
    if gpu_monitor._visible_devices is None:
        logger.info("GPU filtering: DISABLED")
        logger.info("Reason: CUDA_VISIBLE_DEVICES not set or parsing failed")
        logger.info(f"Visible GPUs: ALL ({total_system_gpus} total system GPUs)")
        logger.info("GPU IDs that will be shown: all available GPU IDs")
    elif len(gpu_monitor._visible_devices) == 0:
        logger.info("GPU filtering: ENABLED")
        logger.info("Reason: CUDA_VISIBLE_DEVICES set to empty string")
        logger.info(f"Visible GPUs: NONE (0 of {total_system_gpus} total system GPUs)")
        logger.info("GPU IDs that will be shown: []")
    else:
        logger.info("GPU filtering: ENABLED")
        logger.info(f"Reason: CUDA_VISIBLE_DEVICES specifies specific GPU indices")
        logger.info(f"Visible GPUs: {len(gpu_monitor._visible_devices)} of {total_system_gpus} total system GPUs")
        logger.info(f"GPU IDs that will be shown: {gpu_monitor._visible_devices}")
    
    # Log server configuration
    logger.info("-" * 40)
    logger.info("Server Configuration:")
    logger.info(f"  Host: {HOST}")
    logger.info(f"  Port: {PORT}")
    logger.info(f"  Reload: {RELOAD}")
    logger.info(f"  Log Level: {LOG_LEVEL}")
    logger.info(f"  Refresh Interval: {REFRESH_INTERVAL}s")
    logger.info(f"  CORS Enabled: {ENABLE_CORS}")
    logger.info("-" * 40)
    
    # Final startup message
    if gpu_monitor._visible_devices is not None and len(gpu_monitor._visible_devices) != total_system_gpus:
        logger.info(f"Server will report statistics for {len(gpu_monitor._visible_devices) if gpu_monitor._visible_devices else 0} filtered GPUs")
    else:
        logger.info(f"Server will report statistics for all {total_system_gpus} system GPUs")
    
    logger.info("Starting FastAPI server with Uvicorn...")
    
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
