"""GPU resource state management for tracking statistics and assignments."""

import threading
from typing import Optional, Dict, List
from datetime import datetime

from .models import GPUStats, GPUInfo, WorkerInfo, GPUAssignment


class GPUResourceState:
    """Thread-safe GPU resource state tracking with stats and assignment management."""
    
    def __init__(self):
        """Initialize GPU resource state with thread-safe operations."""
        self._lock = threading.RLock()
        self._gpu_stats: Optional[GPUStats] = None
        self._assignments: Dict[int, List[WorkerInfo]] = {}
        self._last_update: Optional[datetime] = None
    
    def update_stats(self, stats: GPUStats) -> None:
        """Update GPU statistics in a thread-safe manner.
        
        Args:
            stats: New GPU statistics to update
        """
        with self._lock:
            self._gpu_stats = stats
            self._last_update = datetime.now()
            
            # Clean up assignments for GPUs that no longer exist
            if stats.gpus_summary:
                existing_gpu_ids = {gpu.gpu_id for gpu in stats.gpus_summary}
                gpu_ids_to_remove = set(self._assignments.keys()) - existing_gpu_ids
                for gpu_id in gpu_ids_to_remove:
                    del self._assignments[gpu_id]
    
    def get_current_stats(self) -> Optional[GPUStats]:
        """Get current GPU statistics in a thread-safe manner.
        
        Returns:
            Current GPU statistics or None if not available
        """
        with self._lock:
            return self._gpu_stats
    
    def get_last_update(self) -> Optional[datetime]:
        """Get timestamp of last statistics update.
        
        Returns:
            Datetime of last update or None if never updated
        """
        with self._lock:
            return self._last_update
    
    def get_available_gpus(self) -> List[GPUInfo]:
        """Get list of all GPUs from current statistics.
        
        Returns:
            List of GPUInfo objects, empty if no stats available
        """
        with self._lock:
            if self._gpu_stats is None:
                return []
            return self._gpu_stats.gpus_summary.copy()
    
    def get_assignments(self, gpu_id: int) -> List[WorkerInfo]:
        """Get list of workers assigned to a specific GPU.
        
        Args:
            gpu_id: ID of the GPU to query
            
        Returns:
            List of WorkerInfo objects assigned to the GPU
        """
        with self._lock:
            return self._assignments.get(gpu_id, []).copy()
    
    def get_all_assignments(self) -> Dict[int, List[WorkerInfo]]:
        """Get all current GPU assignments.
        
        Returns:
            Dictionary mapping GPU IDs to lists of assigned workers
        """
        with self._lock:
            return {gpu_id: workers.copy() for gpu_id, workers in self._assignments.items()}
    
    def add_assignment(self, gpu_id: int, worker: WorkerInfo) -> None:
        """Add a worker assignment to a GPU in a thread-safe manner.
        
        Args:
            gpu_id: ID of the GPU to assign worker to
            worker: WorkerInfo object to assign
            
        Raises:
            ValueError: If gpu_id is invalid or worker is already assigned to this GPU
        """
        if not isinstance(gpu_id, int) or gpu_id < 0:
            raise ValueError(f"gpu_id must be a non-negative integer, got {gpu_id}")
        
        if not isinstance(worker, WorkerInfo):
            raise ValueError(f"worker must be a WorkerInfo instance, got {type(worker)}")
        
        with self._lock:
            # Initialize assignments list for GPU if it doesn't exist
            if gpu_id not in self._assignments:
                self._assignments[gpu_id] = []
            
            # Check if worker is already assigned to this GPU
            existing_worker_ids = {w.id for w in self._assignments[gpu_id]}
            if worker.id in existing_worker_ids:
                raise ValueError(f"Worker {worker.id} is already assigned to GPU {gpu_id}")
            
            self._assignments[gpu_id].append(worker)
    
    def remove_assignment(self, gpu_id: int, worker_id: str) -> bool:
        """Remove a worker assignment from a GPU in a thread-safe manner.
        
        Args:
            gpu_id: ID of the GPU to remove worker from
            worker_id: ID of the worker to remove
            
        Returns:
            True if worker was found and removed, False otherwise
        """
        if not isinstance(gpu_id, int) or gpu_id < 0:
            raise ValueError(f"gpu_id must be a non-negative integer, got {gpu_id}")
        
        if not isinstance(worker_id, str) or not worker_id.strip():
            raise ValueError(f"worker_id must be a non-empty string, got {worker_id}")
        
        with self._lock:
            if gpu_id not in self._assignments:
                return False
            
            # Find and remove the worker
            workers = self._assignments[gpu_id]
            for i, worker in enumerate(workers):
                if worker.id == worker_id:
                    workers.pop(i)
                    # Clean up empty assignment lists
                    if not workers:
                        del self._assignments[gpu_id]
                    return True
            
            return False
    
    def get_assignment_count(self, gpu_id: int) -> int:
        """Get the number of workers assigned to a specific GPU.
        
        Args:
            gpu_id: ID of the GPU to query
            
        Returns:
            Number of workers assigned to the GPU
        """
        with self._lock:
            return len(self._assignments.get(gpu_id, []))
    
    def get_total_assignments(self) -> int:
        """Get the total number of worker assignments across all GPUs.
        
        Returns:
            Total number of assigned workers
        """
        with self._lock:
            return sum(len(workers) for workers in self._assignments.values())
    
    def clear_assignments(self) -> None:
        """Clear all worker assignments in a thread-safe manner."""
        with self._lock:
            self._assignments.clear()
    
    def clear_all(self) -> None:
        """Clear all state including stats and assignments."""
        with self._lock:
            self._gpu_stats = None
            self._assignments.clear()
            self._last_update = None


class WorkerAssignmentTracker:
    """Thread-safe worker assignment lifecycle management."""
    
    def __init__(self):
        """Initialize worker assignment tracker with thread-safe operations."""
        self._lock = threading.RLock()
        self._assignments: Dict[str, GPUAssignment] = {}
    
    def assign(self, worker_id: str, gpu_id: int) -> GPUAssignment:
        """Create a new worker assignment.
        
        Args:
            worker_id: ID of the worker to assign
            gpu_id: ID of the GPU to assign to
            
        Returns:
            GPUAssignment object representing the assignment
            
        Raises:
            ValueError: If worker is already assigned or parameters are invalid
        """
        if not isinstance(worker_id, str) or not worker_id.strip():
            raise ValueError(f"worker_id must be a non-empty string, got {worker_id}")
        
        if not isinstance(gpu_id, int) or gpu_id < 0:
            raise ValueError(f"gpu_id must be a non-negative integer, got {gpu_id}")
        
        with self._lock:
            if worker_id in self._assignments:
                existing = self._assignments[worker_id]
                raise ValueError(f"Worker {worker_id} is already assigned to GPU {existing.gpu_id}")
            
            assignment = GPUAssignment(
                gpu_id=gpu_id,
                worker_id=worker_id,
                assigned_at=datetime.now()
            )
            
            self._assignments[worker_id] = assignment
            return assignment
    
    def release(self, worker_id: str) -> Optional[GPUAssignment]:
        """Release a worker assignment.
        
        Args:
            worker_id: ID of the worker to release
            
        Returns:
            GPUAssignment that was released, or None if worker wasn't assigned
        """
        if not isinstance(worker_id, str) or not worker_id.strip():
            raise ValueError(f"worker_id must be a non-empty string, got {worker_id}")
        
        with self._lock:
            return self._assignments.pop(worker_id, None)
    
    def get_assignment(self, worker_id: str) -> Optional[GPUAssignment]:
        """Get the current assignment for a worker.
        
        Args:
            worker_id: ID of the worker to query
            
        Returns:
            GPUAssignment if worker is assigned, None otherwise
        """
        if not isinstance(worker_id, str) or not worker_id.strip():
            raise ValueError(f"worker_id must be a non-empty string, got {worker_id}")
        
        with self._lock:
            return self._assignments.get(worker_id)
    
    def get_all_assignments(self) -> List[GPUAssignment]:
        """Get all current worker assignments.
        
        Returns:
            List of all GPUAssignment objects
        """
        with self._lock:
            return list(self._assignments.values())
    
    def get_workers_for_gpu(self, gpu_id: int) -> List[str]:
        """Get list of worker IDs assigned to a specific GPU.
        
        Args:
            gpu_id: ID of the GPU to query
            
        Returns:
            List of worker IDs assigned to the GPU
        """
        if not isinstance(gpu_id, int) or gpu_id < 0:
            raise ValueError(f"gpu_id must be a non-negative integer, got {gpu_id}")
        
        with self._lock:
            return [assignment.worker_id for assignment in self._assignments.values() 
                   if assignment.gpu_id == gpu_id]
    
    def get_assignment_count(self) -> int:
        """Get the total number of active assignments.
        
        Returns:
            Number of currently assigned workers
        """
        with self._lock:
            return len(self._assignments)
    
    def clear_all(self) -> None:
        """Clear all worker assignments."""
        with self._lock:
            self._assignments.clear()
    
    def is_assigned(self, worker_id: str) -> bool:
        """Check if a worker is currently assigned to a GPU.
        
        Args:
            worker_id: ID of the worker to check
            
        Returns:
            True if worker is assigned, False otherwise
        """
        if not isinstance(worker_id, str) or not worker_id.strip():
            raise ValueError(f"worker_id must be a non-empty string, got {worker_id}")
        
        with self._lock:
            return worker_id in self._assignments