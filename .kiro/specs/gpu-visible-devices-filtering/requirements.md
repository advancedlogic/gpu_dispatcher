# Requirements Document

## Introduction

This feature adds support for filtering GPU statistics and summaries based on the CUDA_VISIBLE_DEVICES environment variable in the GPU Statistics Server. Currently, the server reports statistics for all available GPUs on the system, but users often need to restrict GPU visibility to a specific subset of devices for containerized environments, multi-tenant systems, or resource allocation scenarios.

The CUDA_VISIBLE_DEVICES environment variable is a standard NVIDIA CUDA mechanism that allows users to specify which GPUs should be visible to CUDA applications. This feature will respect this setting and only report statistics for the specified GPUs, making the server behavior consistent with CUDA applications.

## Requirements

### Requirement 1

**User Story:** As a system administrator, I want to configure the GPU Statistics Server to only report statistics for specific GPUs using CUDA_VISIBLE_DEVICES, so that I can control GPU visibility in containerized or multi-tenant environments.

#### Acceptance Criteria

1. WHEN CUDA_VISIBLE_DEVICES environment variable is set THEN the server SHALL only report statistics for the specified GPU indices
2. WHEN CUDA_VISIBLE_DEVICES is set to "0,2" THEN the server SHALL only include GPUs 0 and 2 in all API responses
3. WHEN CUDA_VISIBLE_DEVICES is set to "1" THEN the server SHALL only include GPU 1 in all API responses
4. WHEN CUDA_VISIBLE_DEVICES is not set THEN the server SHALL report statistics for all available GPUs (current behavior)
5. WHEN CUDA_VISIBLE_DEVICES is set to an empty string THEN the server SHALL report no GPUs available

### Requirement 2

**User Story:** As a developer, I want the GPU filtering to work consistently across all API endpoints, so that my applications receive consistent GPU information regardless of which endpoint I use.

#### Acceptance Criteria

1. WHEN GPU filtering is active THEN the /gpu/stats endpoint SHALL only return filtered GPUs
2. WHEN GPU filtering is active THEN the /gpu/count endpoint SHALL return the count of filtered GPUs only
3. WHEN GPU filtering is active THEN the /gpu/summary endpoint SHALL calculate summaries based only on filtered GPUs
4. WHEN GPU filtering is active THEN all memory and utilization calculations SHALL be based only on visible GPUs
5. WHEN GPU filtering is active THEN GPU IDs in responses SHALL maintain their original system GPU indices

### Requirement 3

**User Story:** As a system administrator, I want the server to handle invalid CUDA_VISIBLE_DEVICES values gracefully, so that the system remains stable even with configuration errors.

#### Acceptance Criteria

1. WHEN CUDA_VISIBLE_DEVICES contains invalid GPU indices THEN the server SHALL log warnings and skip invalid indices
2. WHEN CUDA_VISIBLE_DEVICES contains non-numeric values THEN the server SHALL log warnings and skip invalid values
3. WHEN CUDA_VISIBLE_DEVICES references GPUs that don't exist THEN the server SHALL log warnings and skip non-existent GPUs
4. WHEN all GPU indices in CUDA_VISIBLE_DEVICES are invalid THEN the server SHALL return empty GPU lists with appropriate error messages
5. WHEN CUDA_VISIBLE_DEVICES parsing fails THEN the server SHALL fall back to showing all GPUs and log the error

### Requirement 4

**User Story:** As a developer, I want to see which GPUs are being filtered in the server configuration and logs, so that I can debug GPU visibility issues.

#### Acceptance Criteria

1. WHEN the server starts THEN it SHALL log the CUDA_VISIBLE_DEVICES value being used
2. WHEN the server starts THEN it SHALL log which specific GPU indices will be visible
3. WHEN the /config endpoint is called THEN it SHALL include the current CUDA_VISIBLE_DEVICES setting
4. WHEN invalid GPU indices are encountered THEN the server SHALL log specific warning messages
5. WHEN GPU filtering is active THEN the server SHALL include filtering information in startup logs

### Requirement 5

**User Story:** As a system administrator, I want the CUDA_VISIBLE_DEVICES filtering to support both comma-separated and space-separated formats, so that it works with different deployment scenarios and follows CUDA conventions.

#### Acceptance Criteria

1. WHEN CUDA_VISIBLE_DEVICES is set to "0,1,2" THEN the server SHALL recognize GPUs 0, 1, and 2
2. WHEN CUDA_VISIBLE_DEVICES is set to "0 1 2" THEN the server SHALL recognize GPUs 0, 1, and 2
3. WHEN CUDA_VISIBLE_DEVICES contains mixed separators like "0,1 2" THEN the server SHALL handle it gracefully
4. WHEN CUDA_VISIBLE_DEVICES contains extra whitespace THEN the server SHALL trim and parse correctly
5. WHEN CUDA_VISIBLE_DEVICES format is invalid THEN the server SHALL log errors and fall back to all GPUs

### Requirement 6

**User Story:** As a developer, I want the GPU filtering to work correctly with the existing caching mechanism, so that performance is maintained while providing accurate filtered results.

#### Acceptance Criteria

1. WHEN GPU filtering is active THEN the caching mechanism SHALL cache only filtered GPU data
2. WHEN CUDA_VISIBLE_DEVICES changes at runtime THEN the cache SHALL be invalidated appropriately
3. WHEN filtered GPU data is cached THEN subsequent requests SHALL return the same filtered results
4. WHEN the refresh interval expires THEN new nvidia-smi calls SHALL respect the current filtering settings
5. WHEN GPU filtering is active THEN cache performance SHALL not be significantly degraded