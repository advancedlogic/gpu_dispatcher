"""Unit tests for GPU allocator functionality."""

import pytest
from datetime import datetime
from unittest.mock import Mock, MagicMock
from gpu_worker_pool.gpu_allocator import ThresholdBasedGPUAllocator
from gpu_worker_pool.models import GPUInfo, GPUStats, WorkerInfo
from gpu_worker_pool.config import ConfigurationManager


class TestThresholdBasedGPUAllocator:
    """Test cases for ThresholdBasedGPUAllocator."""
    
    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration manager."""
        config = Mock(spec=ConfigurationManager)
        config.get_memory_threshold.return_value = 80.0
        config.get_utilization_threshold.return_value = 90.0
        return config
    
    @pytest.fixture
    def allocator(self, mock_config):
        """Create a GPU allocator instance."""
        return ThresholdBasedGPUAllocator(mock_config)
    
    @pytest.fixture
    def sample_gpu_info(self):
        """Create sample GPU info for testing."""
        return [
            GPUInfo(gpu_id=0, name="GPU-0", memory_usage_percent=50.0, utilization_percent=60.0),
            GPUInfo(gpu_id=1, name="GPU-1", memory_usage_percent=75.0, utilization_percent=85.0),
            GPUInfo(gpu_id=2, name="GPU-2", memory_usage_percent=85.0, utilization_percent=70.0),  # Over memory threshold
            GPUInfo(gpu_id=3, name="GPU-3", memory_usage_percent=70.0, utilization_percent=95.0),  # Over util threshold
        ]
    
    @pytest.fixture
    def sample_gpu_stats(self, sample_gpu_info):
        """Create sample GPU stats for testing."""
        return GPUStats(
            gpu_count=4,
            total_memory_mb=32768,
            total_used_memory_mb=16384,
            average_utilization_percent=77.5,
            gpus_summary=sample_gpu_info,
            total_memory_usage_percent=50.0,
            timestamp="2024-01-01T12:00:00Z"
        )
    
    @pytest.fixture
    def sample_worker(self):
        """Create a sample worker for testing."""
        return WorkerInfo(
            id="worker-1",
            enqueued_at=datetime.now(),
            callback=Mock(),
            on_error=Mock()
        )
    
    def test_initialization(self, mock_config):
        """Test allocator initialization with configuration."""
        allocator = ThresholdBasedGPUAllocator(mock_config)
        
        assert allocator._memory_threshold == 80.0
        assert allocator._utilization_threshold == 90.0
        mock_config.get_memory_threshold.assert_called_once()
        mock_config.get_utilization_threshold.assert_called_once()
    
    def test_is_gpu_available_within_thresholds(self, allocator, sample_gpu_info):
        """Test GPU availability when within thresholds."""
        gpu = sample_gpu_info[0]  # 50% memory, 60% utilization
        
        result = allocator.is_gpu_available(gpu, [])
        
        assert result is True
    
    def test_is_gpu_available_memory_threshold_exceeded(self, allocator, sample_gpu_info):
        """Test GPU availability when memory threshold is exceeded."""
        gpu = sample_gpu_info[2]  # 85% memory (over 80% threshold)
        
        result = allocator.is_gpu_available(gpu, [])
        
        assert result is False
    
    def test_is_gpu_available_utilization_threshold_exceeded(self, allocator, sample_gpu_info):
        """Test GPU availability when utilization threshold is exceeded."""
        gpu = sample_gpu_info[3]  # 95% utilization (over 90% threshold)
        
        result = allocator.is_gpu_available(gpu, [])
        
        assert result is False
    
    def test_is_gpu_available_at_threshold_boundary(self, allocator):
        """Test GPU availability at exact threshold boundaries."""
        # GPU at memory threshold (should be unavailable)
        gpu_at_memory_threshold = GPUInfo(
            gpu_id=0, name="GPU-0", 
            memory_usage_percent=80.0, utilization_percent=50.0
        )
        assert allocator.is_gpu_available(gpu_at_memory_threshold, []) is False
        
        # GPU at utilization threshold (should be unavailable)
        gpu_at_util_threshold = GPUInfo(
            gpu_id=1, name="GPU-1", 
            memory_usage_percent=50.0, utilization_percent=90.0
        )
        assert allocator.is_gpu_available(gpu_at_util_threshold, []) is False
        
        # GPU just below thresholds (should be available)
        gpu_below_thresholds = GPUInfo(
            gpu_id=2, name="GPU-2", 
            memory_usage_percent=79.9, utilization_percent=89.9
        )
        assert allocator.is_gpu_available(gpu_below_thresholds, []) is True
    
    def test_calculate_gpu_score(self, allocator, sample_gpu_info):
        """Test GPU score calculation."""
        gpu = sample_gpu_info[0]  # 50% memory, 60% utilization
        
        score = allocator.calculate_gpu_score(gpu)
        
        assert score == 110.0  # 50 + 60
    
    def test_calculate_gpu_score_different_values(self, allocator):
        """Test GPU score calculation with different values."""
        gpu1 = GPUInfo(gpu_id=0, name="GPU-0", memory_usage_percent=30.0, utilization_percent=40.0)
        gpu2 = GPUInfo(gpu_id=1, name="GPU-1", memory_usage_percent=70.0, utilization_percent=20.0)
        
        score1 = allocator.calculate_gpu_score(gpu1)
        score2 = allocator.calculate_gpu_score(gpu2)
        
        assert score1 == 70.0  # 30 + 40
        assert score2 == 90.0  # 70 + 20
        assert score1 < score2  # Lower score is better
    
    def test_find_available_gpu_selects_best(self, allocator, sample_gpu_stats):
        """Test that find_available_gpu selects the GPU with lowest score."""
        assignments = {}
        
        result = allocator.find_available_gpu(sample_gpu_stats, assignments)
        
        # Should select GPU 0 (score: 50+60=110) over GPU 1 (score: 75+85=160)
        # GPUs 2 and 3 are over thresholds and unavailable
        assert result == 0
    
    def test_find_available_gpu_no_available_gpus(self, allocator):
        """Test find_available_gpu when no GPUs are available."""
        # All GPUs over thresholds
        gpu_stats = GPUStats(
            gpu_count=2,
            total_memory_mb=16384,
            total_used_memory_mb=8192,
            average_utilization_percent=95.0,
            gpus_summary=[
                GPUInfo(gpu_id=0, name="GPU-0", memory_usage_percent=85.0, utilization_percent=95.0),
                GPUInfo(gpu_id=1, name="GPU-1", memory_usage_percent=90.0, utilization_percent=92.0),
            ],
            total_memory_usage_percent=87.5,
            timestamp="2024-01-01T12:00:00Z"
        )
        
        result = allocator.find_available_gpu(gpu_stats, {})
        
        assert result is None
    
    def test_find_available_gpu_empty_stats(self, allocator):
        """Test find_available_gpu with empty GPU stats."""
        result = allocator.find_available_gpu(None, {})
        assert result is None
        
        # Empty GPU summary
        empty_stats = GPUStats(
            gpu_count=0,
            total_memory_mb=0,
            total_used_memory_mb=0,
            average_utilization_percent=0.0,
            gpus_summary=[],
            total_memory_usage_percent=0.0,
            timestamp="2024-01-01T12:00:00Z"
        )
        
        result = allocator.find_available_gpu(empty_stats, {})
        assert result is None
    
    def test_find_available_gpu_with_assignments(self, allocator, sample_gpu_stats, sample_worker):
        """Test find_available_gpu considers existing assignments."""
        # Assignments don't affect threshold checking in current implementation
        # but should be passed to is_gpu_available method
        assignments = {0: [sample_worker]}
        
        result = allocator.find_available_gpu(sample_gpu_stats, assignments)
        
        # Should still select GPU 0 as it's within thresholds
        assert result == 0
    
    def test_find_available_gpu_multiple_available_selects_lowest_score(self, allocator):
        """Test that multiple available GPUs are ranked by score."""
        gpu_stats = GPUStats(
            gpu_count=3,
            total_memory_mb=24576,
            total_used_memory_mb=8192,
            average_utilization_percent=50.0,
            gpus_summary=[
                GPUInfo(gpu_id=0, name="GPU-0", memory_usage_percent=60.0, utilization_percent=70.0),  # Score: 130
                GPUInfo(gpu_id=1, name="GPU-1", memory_usage_percent=40.0, utilization_percent=50.0),  # Score: 90 (best)
                GPUInfo(gpu_id=2, name="GPU-2", memory_usage_percent=70.0, utilization_percent=80.0),  # Score: 150
            ],
            total_memory_usage_percent=33.3,
            timestamp="2024-01-01T12:00:00Z"
        )
        
        result = allocator.find_available_gpu(gpu_stats, {})
        
        assert result == 1  # GPU with lowest score (90)
    
    def test_find_available_gpu_edge_case_equal_scores(self, allocator):
        """Test GPU selection when multiple GPUs have equal scores."""
        gpu_stats = GPUStats(
            gpu_count=2,
            total_memory_mb=16384,
            total_used_memory_mb=4096,
            average_utilization_percent=50.0,
            gpus_summary=[
                GPUInfo(gpu_id=0, name="GPU-0", memory_usage_percent=30.0, utilization_percent=40.0),  # Score: 70
                GPUInfo(gpu_id=1, name="GPU-1", memory_usage_percent=35.0, utilization_percent=35.0),  # Score: 70
            ],
            total_memory_usage_percent=25.0,
            timestamp="2024-01-01T12:00:00Z"
        )
        
        result = allocator.find_available_gpu(gpu_stats, {})
        
        # Should return one of them (min() will return the first one with lowest score)
        assert result in [0, 1]
    
    def test_threshold_configuration_affects_availability(self, sample_gpu_info):
        """Test that different threshold configurations affect GPU availability."""
        # Create allocator with stricter thresholds
        strict_config = Mock(spec=ConfigurationManager)
        strict_config.get_memory_threshold.return_value = 60.0
        strict_config.get_utilization_threshold.return_value = 70.0
        strict_allocator = ThresholdBasedGPUAllocator(strict_config)
        
        # Create allocator with lenient thresholds
        lenient_config = Mock(spec=ConfigurationManager)
        lenient_config.get_memory_threshold.return_value = 95.0
        lenient_config.get_utilization_threshold.return_value = 98.0
        lenient_allocator = ThresholdBasedGPUAllocator(lenient_config)
        
        gpu = sample_gpu_info[1]  # 75% memory, 85% utilization
        
        # Strict allocator should reject this GPU
        assert strict_allocator.is_gpu_available(gpu, []) is False
        
        # Lenient allocator should accept this GPU
        assert lenient_allocator.is_gpu_available(gpu, []) is True

    def test_gpu_selection_with_assignment_tracking(self, allocator, sample_worker):
        """Test GPU selection scenarios with assignment tracking integration."""
        # Create GPU stats with multiple available GPUs
        gpu_stats = GPUStats(
            gpu_count=3,
            total_memory_mb=24576,
            total_used_memory_mb=8192,
            average_utilization_percent=50.0,
            gpus_summary=[
                GPUInfo(gpu_id=0, name="GPU-0", memory_usage_percent=40.0, utilization_percent=30.0),  # Score: 70
                GPUInfo(gpu_id=1, name="GPU-1", memory_usage_percent=50.0, utilization_percent=40.0),  # Score: 90
                GPUInfo(gpu_id=2, name="GPU-2", memory_usage_percent=30.0, utilization_percent=35.0),  # Score: 65 (best)
            ],
            total_memory_usage_percent=33.3,
            timestamp="2024-01-01T12:00:00Z"
        )
        
        # Test with no assignments
        result = allocator.find_available_gpu(gpu_stats, {})
        assert result == 2  # GPU with lowest score
        
        # Test with existing assignments (should still work the same way)
        assignments = {
            0: [sample_worker],
            1: [],
            2: []
        }
        result = allocator.find_available_gpu(gpu_stats, assignments)
        assert result == 2  # Still selects GPU with lowest score
    
    def test_gpu_selection_lowest_combined_resource_usage(self, allocator):
        """Test that GPU selection prioritizes lowest combined resource usage."""
        gpu_stats = GPUStats(
            gpu_count=4,
            total_memory_mb=32768,
            total_used_memory_mb=8192,
            average_utilization_percent=50.0,
            gpus_summary=[
                GPUInfo(gpu_id=0, name="GPU-0", memory_usage_percent=70.0, utilization_percent=10.0),  # Score: 80
                GPUInfo(gpu_id=1, name="GPU-1", memory_usage_percent=20.0, utilization_percent=50.0),  # Score: 70
                GPUInfo(gpu_id=2, name="GPU-2", memory_usage_percent=45.0, utilization_percent=30.0),  # Score: 75
                GPUInfo(gpu_id=3, name="GPU-3", memory_usage_percent=10.0, utilization_percent=55.0),  # Score: 65 (best)
            ],
            total_memory_usage_percent=25.0,
            timestamp="2024-01-01T12:00:00Z"
        )
        
        result = allocator.find_available_gpu(gpu_stats, {})
        assert result == 3  # GPU with lowest combined resource usage (65)
    
    def test_gpu_selection_scenarios_with_thresholds(self, allocator):
        """Test various GPU selection scenarios respecting thresholds."""
        # Scenario 1: Some GPUs over threshold, select from available ones
        gpu_stats = GPUStats(
            gpu_count=4,
            total_memory_mb=32768,
            total_used_memory_mb=16384,
            average_utilization_percent=70.0,
            gpus_summary=[
                GPUInfo(gpu_id=0, name="GPU-0", memory_usage_percent=85.0, utilization_percent=60.0),  # Over memory threshold
                GPUInfo(gpu_id=1, name="GPU-1", memory_usage_percent=70.0, utilization_percent=95.0),  # Over util threshold
                GPUInfo(gpu_id=2, name="GPU-2", memory_usage_percent=60.0, utilization_percent=70.0),  # Score: 130
                GPUInfo(gpu_id=3, name="GPU-3", memory_usage_percent=50.0, utilization_percent=60.0),  # Score: 110 (best available)
            ],
            total_memory_usage_percent=50.0,
            timestamp="2024-01-01T12:00:00Z"
        )
        
        result = allocator.find_available_gpu(gpu_stats, {})
        assert result == 3  # Best available GPU within thresholds
        
        # Scenario 2: Only one GPU available
        single_available_stats = GPUStats(
            gpu_count=3,
            total_memory_mb=24576,
            total_used_memory_mb=16384,
            average_utilization_percent=85.0,
            gpus_summary=[
                GPUInfo(gpu_id=0, name="GPU-0", memory_usage_percent=85.0, utilization_percent=95.0),  # Over both thresholds
                GPUInfo(gpu_id=1, name="GPU-1", memory_usage_percent=90.0, utilization_percent=85.0),  # Over memory threshold
                GPUInfo(gpu_id=2, name="GPU-2", memory_usage_percent=75.0, utilization_percent=80.0),  # Available
            ],
            total_memory_usage_percent=66.7,
            timestamp="2024-01-01T12:00:00Z"
        )
        
        result = allocator.find_available_gpu(single_available_stats, {})
        assert result == 2  # Only available GPU
    
    def test_assignment_tracking_integration_detailed(self, allocator, sample_worker):
        """Test detailed integration with assignment tracking."""
        # Create additional workers for testing
        worker2 = WorkerInfo(
            id="worker-2",
            enqueued_at=datetime.now(),
            callback=Mock(),
            on_error=Mock()
        )
        
        gpu_stats = GPUStats(
            gpu_count=2,
            total_memory_mb=16384,
            total_used_memory_mb=4096,
            average_utilization_percent=40.0,
            gpus_summary=[
                GPUInfo(gpu_id=0, name="GPU-0", memory_usage_percent=30.0, utilization_percent=40.0),  # Score: 70
                GPUInfo(gpu_id=1, name="GPU-1", memory_usage_percent=35.0, utilization_percent=30.0),  # Score: 65 (better)
            ],
            total_memory_usage_percent=25.0,
            timestamp="2024-01-01T12:00:00Z"
        )
        
        # Test with different assignment scenarios
        assignments_scenarios = [
            {},  # No assignments
            {0: []},  # Empty assignment list for GPU 0
            {0: [sample_worker]},  # One worker assigned to GPU 0
            {0: [sample_worker], 1: [worker2]},  # Workers assigned to both GPUs
            {1: [sample_worker, worker2]},  # Multiple workers on GPU 1
        ]
        
        for assignments in assignments_scenarios:
            result = allocator.find_available_gpu(gpu_stats, assignments)
            # Should always select GPU 1 as it has the lower score (65 vs 70)
            # Assignment tracking is passed but doesn't affect threshold evaluation in current implementation
            assert result == 1


if __name__ == "__main__":
    pytest.main([__file__])