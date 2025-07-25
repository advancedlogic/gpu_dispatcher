# Implementation Plan

- [x] 1. Implement CUDA_VISIBLE_DEVICES parsing functionality
  - Add `_parse_visible_devices()` method to GPUMonitor class that parses CUDA_VISIBLE_DEVICES environment variable
  - Support comma-separated format ("0,1,2"), space-separated format ("0 1 2"), and mixed separators
  - Handle empty string to show no GPUs, None/missing to show all GPUs
  - Include robust error handling for invalid values with fallback to all GPUs
  - Add comprehensive logging for debugging invalid values and parsing results
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 3.1, 3.2, 3.5, 5.1, 5.2, 5.3, 5.4, 5.5_

- [x] 2. Implement GPU data filtering logic
  - Add `_filter_gpu_data()` method that filters nvidia-smi results based on visible devices
  - Maintain original GPU IDs in filtered results while excluding non-visible GPUs
  - Handle cases where CUDA_VISIBLE_DEVICES references non-existent GPUs with warnings
  - Update gpu_count to reflect only filtered GPUs
  - Preserve all GPU metadata and timestamp information in filtered results
  - _Requirements: 2.1, 2.5, 3.1, 3.3, 3.4_

- [x] 3. Integrate filtering into GPUMonitor initialization and data fetching
  - Initialize `_visible_devices` in GPUMonitor.__init__() by calling `_parse_visible_devices()`
  - Modify `_fetch_gpu_stats()` to apply filtering after nvidia-smi data collection
  - Ensure filtering works with existing caching mechanism without performance degradation
  - Add startup logging to show CUDA_VISIBLE_DEVICES value and visible GPU IDs
  - _Requirements: 4.1, 4.2, 6.1, 6.3, 6.4, 6.5_

- [x] 4. Update API endpoints to ensure consistent filtering
  - Verify `/gpu/stats` endpoint returns only filtered GPUs in response
  - Verify `/gpu/count` endpoint returns count of filtered GPUs only
  - Verify `/gpu/summary` endpoint calculates summaries using only filtered GPUs
  - Ensure all memory and utilization calculations are based on visible GPUs only
  - Test that all endpoints return consistent GPU sets when filtering is active
  - _Requirements: 2.1, 2.2, 2.3, 2.4_

- [x] 5. Enhance configuration endpoint with filtering information
  - Add new `gpu_filtering` section to `/config` endpoint response
  - Include `cuda_visible_devices`, `visible_gpu_ids`, and `filtering_active` fields
  - Add `total_system_gpus` and `visible_gpu_count` for debugging information
  - Ensure configuration endpoint shows current filtering state accurately
  - _Requirements: 4.3_

- [x] 6. Add comprehensive error handling and logging
  - Implement warning logs for invalid GPU indices in CUDA_VISIBLE_DEVICES
  - Add error logging when CUDA_VISIBLE_DEVICES parsing fails completely
  - Log specific warnings when referenced GPUs don't exist on the system
  - Ensure graceful fallback to all GPUs when parsing errors occur
  - Add informative startup logs showing filtering configuration
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 4.1, 4.2, 4.4_

- [x] 7. Create unit tests for CUDA_VISIBLE_DEVICES parsing
  - Test parsing of comma-separated values ("0,1,2")
  - Test parsing of space-separated values ("0 1 2")
  - Test parsing of mixed separators ("0,1 2")
  - Test handling of empty string and None values
  - Test error handling for invalid values ("abc", "-1", mixed valid/invalid)
  - Test whitespace handling and edge cases
  - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5_

- [x] 8. Create unit tests for GPU data filtering
  - Test filtering with valid GPU IDs that exist on system
  - Test filtering with non-existent GPU IDs and verify warnings
  - Test filtering with empty visible devices list (no GPUs shown)
  - Test filtering disabled (None) shows all GPUs
  - Test filtering with error responses from nvidia-smi passes through errors
  - Verify filtered data maintains correct structure and GPU IDs
  - _Requirements: 3.1, 3.3, 3.4_

- [x] 9. Create integration tests for API endpoint consistency
  - Test that all endpoints (/gpu/stats, /gpu/count, /gpu/summary) return consistent filtered GPU sets
  - Verify GPU counts match across all endpoints when filtering is active
  - Test that summary calculations (memory totals, averages) use only filtered GPUs
  - Test configuration endpoint returns correct filtering information
  - Verify caching works correctly with filtered data
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 6.1, 6.2, 6.3_

- [x] 10. Update documentation for CUDA_VISIBLE_DEVICES filtering
  - Add CUDA_VISIBLE_DEVICES configuration section to SERVER.md
  - Document supported formats and examples in SERVER.md
  - Update environment variables table with CUDA_VISIBLE_DEVICES
  - Add troubleshooting section for filtering issues
  - Update API.md with new configuration endpoint fields
  - Add examples showing filtered vs unfiltered responses
  - _Requirements: 4.3_