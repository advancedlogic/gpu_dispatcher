"""GPU allocation logic for the Worker Pool system."""

import logging
from abc import ABC, abstractmethod
from typing import Optional, Dict, List
from .models import GPUStats, GPUInfo, WorkerInfo
from .config import ConfigurationManager

logger = logging.getLogger(__name__)


class GPUAllocator(ABC):
    """Abstract base class for GPU allocation logic."""
    
    @abstractmethod
    def find_available_gpu(self, current_stats: GPUStats, assignments: Dict[int, List[WorkerInfo]]) -> Optional[int]:
        """Find the best available GPU for assignment."""
        pass
    
    @abstractmethod
    def is_gpu_available(self, gpu: GPUInfo, assigned_workers: List[WorkerInfo]) -> bool:
        """Check if a GPU is available based on thresholds."""
        pass
    
    @abstractmethod
    def calculate_gpu_score(self, gpu: GPUInfo) -> float:
        """Calculate a score for GPU selection (lower is better)."""
        pass


class ThresholdBasedGPUAllocator(GPUAllocator):
    """GPU allocator that uses configurable thresholds for availability evaluation."""
    
    def __init__(self, config: ConfigurationManager):
        """Initialize the allocator with configuration."""
        self._config = config
        self._memory_threshold = config.get_memory_threshold()
        self._utilization_threshold = config.get_utilization_threshold()
        
        logger.info(
            f"GPUAllocator initialized with thresholds: "
            f"memory={self._memory_threshold}%, utilization={self._utilization_threshold}%"
        )
    
    def find_available_gpu(self, current_stats: GPUStats, assignments: Dict[int, List[WorkerInfo]]) -> Optional[int]:
        """
        Find the best available GPU for assignment.
        
        Evaluates all GPUs against thresholds and selects the one with the lowest
        combined resource usage (memory + utilization).
        
        Args:
            current_stats: Current GPU statistics from monitoring service
            assignments: Current worker assignments per GPU
            
        Returns:
            GPU ID of the best available GPU, or None if no GPUs are available
        """
        if not current_stats or not current_stats.gpus_summary:
            logger.warning("No GPU statistics available for allocation")
            return None
        
        available_gpus = []
        
        # Evaluate each GPU for availability
        for gpu in current_stats.gpus_summary:
            assigned_workers = assignments.get(gpu.gpu_id, [])
            
            if self.is_gpu_available(gpu, assigned_workers):
                score = self.calculate_gpu_score(gpu)
                available_gpus.append((gpu.gpu_id, score))
                logger.debug(
                    f"GPU {gpu.gpu_id} available with score {score:.2f} "
                    f"(memory: {gpu.memory_usage_percent}%, util: {gpu.utilization_percent}%)"
                )
            else:
                logger.debug(
                    f"GPU {gpu.gpu_id} unavailable "
                    f"(memory: {gpu.memory_usage_percent}%, util: {gpu.utilization_percent}%)"
                )
        
        if not available_gpus:
            logger.info("No GPUs available for assignment")
            return None
        
        # Select GPU with lowest score (best combined resource usage)
        best_gpu_id, best_score = min(available_gpus, key=lambda x: x[1])
        
        logger.info(
            f"Selected GPU {best_gpu_id} for assignment with score {best_score:.2f}"
        )
        
        return best_gpu_id
    
    def is_gpu_available(self, gpu: GPUInfo, assigned_workers: List[WorkerInfo]) -> bool:
        """
        Check if a GPU is available based on resource thresholds.
        
        A GPU is considered available if both its memory usage and utilization
        are below the configured thresholds.
        
        Args:
            gpu: GPU information including current resource usage
            assigned_workers: List of workers currently assigned to this GPU
            
        Returns:
            True if GPU is available for assignment, False otherwise
        """
        # Check memory threshold
        if gpu.memory_usage_percent >= self._memory_threshold:
            logger.debug(
                f"GPU {gpu.gpu_id} memory usage {gpu.memory_usage_percent}% "
                f"exceeds threshold {self._memory_threshold}%"
            )
            return False
        
        # Check utilization threshold
        if gpu.utilization_percent >= self._utilization_threshold:
            logger.debug(
                f"GPU {gpu.gpu_id} utilization {gpu.utilization_percent}% "
                f"exceeds threshold {self._utilization_threshold}%"
            )
            return False
        
        return True
    
    def calculate_gpu_score(self, gpu: GPUInfo) -> float:
        """
        Calculate a score for GPU selection based on combined resource usage.
        
        The score is calculated as the sum of memory usage percentage and
        utilization percentage. Lower scores indicate better availability.
        
        Args:
            gpu: GPU information including current resource usage
            
        Returns:
            Combined resource usage score (lower is better)
        """
        score = gpu.memory_usage_percent + gpu.utilization_percent
        
        logger.debug(
            f"GPU {gpu.gpu_id} score: {score:.2f} "
            f"(memory: {gpu.memory_usage_percent}% + util: {gpu.utilization_percent}%)"
        )
        
        return score