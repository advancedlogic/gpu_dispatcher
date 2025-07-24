"""Worker queue management for the GPU Worker Pool system."""

import threading
import logging
from collections import deque
from typing import Optional, List
from abc import ABC, abstractmethod
from datetime import datetime

from .models import WorkerInfo


class WorkerQueue(ABC):
    """Abstract base class for worker queue implementations."""
    
    @abstractmethod
    def enqueue(self, worker: WorkerInfo) -> None:
        """Add a worker to the queue."""
        pass
    
    @abstractmethod
    def dequeue(self) -> Optional[WorkerInfo]:
        """Remove and return the next worker from the queue."""
        pass
    
    @abstractmethod
    def size(self) -> int:
        """Return the current size of the queue."""
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """Remove all workers from the queue."""
        pass
    
    @abstractmethod
    def block_worker(self, worker: WorkerInfo, reason: str) -> None:
        """Block a worker by adding it to the queue with logging."""
        pass
    
    @abstractmethod
    def unblock_next_worker(self) -> Optional[WorkerInfo]:
        """Unblock the next worker in the queue with logging."""
        pass
    
    @abstractmethod
    def unblock_workers(self, count: int) -> List[WorkerInfo]:
        """Unblock multiple workers from the queue."""
        pass


class FIFOWorkerQueue(WorkerQueue):
    """Thread-safe FIFO implementation of the worker queue."""
    
    def __init__(self):
        """Initialize the FIFO worker queue."""
        self._queue: deque[WorkerInfo] = deque()
        self._lock = threading.RLock()
    
    def enqueue(self, worker: WorkerInfo) -> None:
        """Add a worker to the end of the queue.
        
        Args:
            worker: The WorkerInfo instance to add to the queue
            
        Raises:
            ValueError: If worker is None or not a WorkerInfo instance
        """
        if worker is None:
            raise ValueError("Worker cannot be None")
        
        if not isinstance(worker, WorkerInfo):
            raise ValueError(f"Worker must be a WorkerInfo instance, got {type(worker)}")
        
        with self._lock:
            self._queue.append(worker)
    
    def dequeue(self) -> Optional[WorkerInfo]:
        """Remove and return the worker from the front of the queue.
        
        Returns:
            The next WorkerInfo instance in the queue, or None if queue is empty
        """
        with self._lock:
            if not self._queue:
                return None
            return self._queue.popleft()
    
    def size(self) -> int:
        """Return the current number of workers in the queue.
        
        Returns:
            The number of workers currently in the queue
        """
        with self._lock:
            return len(self._queue)
    
    def clear(self) -> None:
        """Remove all workers from the queue.
        
        This method will call the on_error callback for each worker
        with a cancellation exception to notify them that they were
        removed from the queue.
        """
        with self._lock:
            # Notify all workers that they are being cancelled
            while self._queue:
                worker = self._queue.popleft()
                try:
                    worker.on_error(Exception("Worker request cancelled due to queue clear"))
                except Exception:
                    # Ignore errors in error callbacks to prevent cascading failures
                    pass
    
    def peek(self) -> Optional[WorkerInfo]:
        """Return the next worker without removing it from the queue.
        
        Returns:
            The next WorkerInfo instance in the queue, or None if queue is empty
        """
        with self._lock:
            if not self._queue:
                return None
            return self._queue[0]
    
    def get_all_workers(self) -> List[WorkerInfo]:
        """Return a copy of all workers currently in the queue.
        
        Returns:
            A list containing all WorkerInfo instances currently in the queue
        """
        with self._lock:
            return list(self._queue)
    
    def remove_worker(self, worker_id: str) -> bool:
        """Remove a specific worker from the queue by ID.
        
        Args:
            worker_id: The ID of the worker to remove
            
        Returns:
            True if the worker was found and removed, False otherwise
        """
        if not worker_id:
            return False
        
        with self._lock:
            for i, worker in enumerate(self._queue):
                if worker.id == worker_id:
                    del self._queue[i]
                    try:
                        worker.on_error(Exception(f"Worker {worker_id} removed from queue"))
                    except Exception:
                        # Ignore errors in error callbacks
                        pass
                    return True
            return False
    
    def block_worker(self, worker: WorkerInfo, reason: str) -> None:
        """Block a worker by adding it to the queue with logging.
        
        Args:
            worker: The WorkerInfo instance to block
            reason: The reason why the worker is being blocked
            
        Raises:
            ValueError: If worker is None or not a WorkerInfo instance
        """
        if worker is None:
            raise ValueError("Worker cannot be None")
        
        if not isinstance(worker, WorkerInfo):
            raise ValueError(f"Worker must be a WorkerInfo instance, got {type(worker)}")
        
        if not reason:
            reason = "No reason provided"
        
        # Log the blocking event
        logger = logging.getLogger(__name__)
        logger.info(
            f"Blocking worker {worker.id} at {datetime.now().isoformat()}: {reason}"
        )
        
        # Add worker to the queue (this blocks them until they can be unblocked)
        with self._lock:
            self._queue.append(worker)
    
    def unblock_next_worker(self) -> Optional[WorkerInfo]:
        """Unblock the next worker in the queue with logging.
        
        Returns:
            The next WorkerInfo instance that was unblocked, or None if queue is empty
        """
        with self._lock:
            if not self._queue:
                return None
            
            worker = self._queue.popleft()
            
            # Log the unblocking event
            logger = logging.getLogger(__name__)
            logger.info(
                f"Unblocking worker {worker.id} at {datetime.now().isoformat()}"
            )
            
            return worker
    
    def unblock_workers(self, count: int) -> List[WorkerInfo]:
        """Unblock multiple workers from the queue.
        
        Args:
            count: Maximum number of workers to unblock
            
        Returns:
            List of WorkerInfo instances that were unblocked
            
        Raises:
            ValueError: If count is negative
        """
        if count < 0:
            raise ValueError(f"Count must be non-negative, got {count}")
        
        if count == 0:
            return []
        
        unblocked_workers = []
        logger = logging.getLogger(__name__)
        
        with self._lock:
            for _ in range(min(count, len(self._queue))):
                if not self._queue:
                    break
                
                worker = self._queue.popleft()
                unblocked_workers.append(worker)
                
                # Log each unblocking event
                logger.info(
                    f"Unblocking worker {worker.id} at {datetime.now().isoformat()}"
                )
        
        if unblocked_workers:
            logger.info(f"Unblocked {len(unblocked_workers)} workers")
        
        return unblocked_workers