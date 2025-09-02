"""
Optimized version of ubatch utilities for better performance with large batch sizes.
"""
import torch
from typing import Optional
from dataclasses import dataclass


@dataclass
class UbatchWorkloadInfo:
    """Information about workload characteristics for intelligent ubatch splitting."""
    total_requests: int
    total_tokens: int
    decode_requests: int
    prefill_requests: int
    decode_tokens: int
    prefill_tokens: int
    query_lens: torch.Tensor
    compute_complexities: torch.Tensor


def analyze_workload(
    query_lens: torch.Tensor,
    seq_lens: torch.Tensor,
    decode_threshold: int = 1
) -> UbatchWorkloadInfo:
    """Fast workload analysis using vectorized operations."""
    num_requests = len(query_lens)
    total_tokens = int(torch.sum(query_lens))
    
    # Vectorized classification
    is_decode = query_lens <= decode_threshold
    decode_requests = int(torch.sum(is_decode))
    prefill_requests = num_requests - decode_requests
    
    decode_tokens = int(torch.sum(query_lens[is_decode]))
    prefill_tokens = total_tokens - decode_tokens
    
    # Vectorized complexity computation
    compute_complexities = query_lens.float() * seq_lens.float()
    
    return UbatchWorkloadInfo(
        total_requests=num_requests,
        total_tokens=total_tokens,
        decode_requests=decode_requests,
        prefill_requests=prefill_requests,
        decode_tokens=decode_tokens,
        prefill_tokens=prefill_tokens,
        query_lens=query_lens,
        compute_complexities=compute_complexities
    )


def create_fast_balanced_ubatch_slices(
    workload_info: UbatchWorkloadInfo,
    num_ubatches: int = 2,
    balance_strategy: str = "compute_complexity"
) -> list:
    """
    Fast ubatch splitting for large batches using greedy heuristics instead of DP.
    
    This version sacrifices optimal balancing for O(n log n) complexity instead of O(n^2 * k).
    """
    from vllm.v1.attention.backends.utils import UbatchSlice
    
    num_requests = workload_info.total_requests
    
    # Handle edge cases
    if num_requests <= num_ubatches:
        return _create_single_request_ubatches_fast(workload_info)
    
    # Use simple approach for small batches
    if num_requests <= 32:
        return _create_simple_consecutive_ubatch_slices_fast(workload_info, num_ubatches)
    
    # For large batches, use fast greedy splitting
    return _create_fast_greedy_ubatch_slices(workload_info, num_ubatches, balance_strategy)


def _create_single_request_ubatches_fast(workload_info: UbatchWorkloadInfo) -> list:
    """Fast creation of single-request ubatches."""
    from vllm.v1.attention.backends.utils import UbatchSlice
    
    ubatch_slices = []
    token_offset = 0
    
    for i in range(workload_info.total_requests):
        request_tokens = int(workload_info.query_lens[i])
        
        ubatch_slice = UbatchSlice(
            request_slice=slice(i, i + 1),
            token_slice=slice(token_offset, token_offset + request_tokens),
            compute_complexity=float(workload_info.compute_complexities[i]),
            query_lens=workload_info.query_lens[i:i+1],
            is_prefill=bool(workload_info.query_lens[i] > 1),
            max_query_len=int(workload_info.query_lens[i])
        )
        ubatch_slices.append(ubatch_slice)
        token_offset += request_tokens
    
    return ubatch_slices


def _create_simple_consecutive_ubatch_slices_fast(
    workload_info: UbatchWorkloadInfo,
    num_ubatches: int
) -> list:
    """Fast creation of consecutive ubatch slices."""
    from vllm.v1.attention.backends.utils import UbatchSlice
    
    num_requests = workload_info.total_requests
    ubatch_slices = []
    
    for i in range(num_ubatches):
        # Calculate request boundaries
        request_start = (i * num_requests) // num_ubatches
        request_end = ((i + 1) * num_requests) // num_ubatches
        
        # Calculate token boundaries efficiently
        token_start = int(torch.sum(workload_info.query_lens[:request_start])) if request_start > 0 else 0
        token_end = int(torch.sum(workload_info.query_lens[:request_end]))
        
        # Slice data efficiently
        slice_query_lens = workload_info.query_lens[request_start:request_end]
        slice_complexities = workload_info.compute_complexities[request_start:request_end]
        
        ubatch_slice = UbatchSlice(
            request_slice=slice(request_start, request_end),
            token_slice=slice(token_start, token_end),
            compute_complexity=float(torch.sum(slice_complexities)),
            query_lens=slice_query_lens,
            is_prefill=bool(torch.any(slice_query_lens > 1)),
            max_query_len=int(torch.max(slice_query_lens))
        )
        ubatch_slices.append(ubatch_slice)
    
    return ubatch_slices


def _create_fast_greedy_ubatch_slices(
    workload_info: UbatchWorkloadInfo,
    num_ubatches: int,
    balance_strategy: str = "compute_complexity"
) -> list:
    """
    Fast greedy splitting for large batches O(n log n) complexity.
    
    Uses a simple greedy approach: sort requests by weight, then distribute
    in round-robin fashion to achieve reasonable balance.
    """
    from vllm.v1.attention.backends.utils import UbatchSlice
    
    # Choose weights based on strategy
    if balance_strategy == "compute_complexity":
        weights = workload_info.compute_complexities
    else:  # "tokens"
        weights = workload_info.query_lens.float()
    
    # Sort requests by weight (heaviest first)
    sorted_indices = torch.argsort(weights, descending=True)
    
    # Distribute requests in round-robin to achieve balance
    ubatch_request_lists = [[] for _ in range(num_ubatches)]
    
    for i, idx in enumerate(sorted_indices):
        ubatch_idx = i % num_ubatches
        ubatch_request_lists[ubatch_idx].append(int(idx))
    
    # Sort each ubatch's requests to maintain consecutive memory access
    for req_list in ubatch_request_lists:
        req_list.sort()
    
    # Create ubatch slices
    ubatch_slices = []
    
    for req_indices in ubatch_request_lists:
        if not req_indices:
            continue
            
        # Convert to consecutive ranges where possible for efficiency
        ranges = _consolidate_to_ranges(req_indices)
        
        if len(ranges) == 1:
            # Single consecutive range - use slice
            start, end = ranges[0]
            
            token_start = int(torch.sum(workload_info.query_lens[:start])) if start > 0 else 0
            token_end = int(torch.sum(workload_info.query_lens[:end]))
            
            slice_query_lens = workload_info.query_lens[start:end]
            slice_complexities = workload_info.compute_complexities[start:end]
            
            ubatch_slice = UbatchSlice(
                request_slice=slice(start, end),
                token_slice=slice(token_start, token_end),
                compute_complexity=float(torch.sum(slice_complexities)),
                query_lens=slice_query_lens,
                is_prefill=bool(torch.any(slice_query_lens > 1)),
                max_query_len=int(torch.max(slice_query_lens))
            )
        else:
            # Multiple ranges - use indices (future extension)
            # For now, create a slice covering the full range and mark non-consecutive
            start, end = min(req_indices), max(req_indices) + 1
            
            token_start = int(torch.sum(workload_info.query_lens[:start])) if start > 0 else 0
            token_end = int(torch.sum(workload_info.query_lens[:end]))
            
            # Calculate actual tokens for this ubatch
            actual_tokens = int(torch.sum(workload_info.query_lens[req_indices]))
            
            # Use the range slice but note it's not fully consecutive
            slice_query_lens = workload_info.query_lens[req_indices]
            slice_complexities = workload_info.compute_complexities[req_indices]
            
            ubatch_slice = UbatchSlice(
                request_slice=slice(start, end),  # Conservative range
                token_slice=slice(token_start, token_start + actual_tokens),
                compute_complexity=float(torch.sum(slice_complexities)),
                query_lens=slice_query_lens,
                is_prefill=bool(torch.any(slice_query_lens > 1)),
                max_query_len=int(torch.max(slice_query_lens)),
                request_indices=torch.tensor(req_indices, dtype=torch.long)
            )
        
        ubatch_slices.append(ubatch_slice)
    
    return ubatch_slices


def _consolidate_to_ranges(indices):
    """
    Convert a list of indices to consecutive ranges.
    
    Args:
        indices: Sorted list of indices
        
    Returns:
        List of (start, end) tuples representing consecutive ranges
    """
    if not indices:
        return []
    
    ranges = []
    current_start = indices[0]
    current_end = indices[0]
    
    for i in range(1, len(indices)):
        if indices[i] == current_end + 1:
            # Consecutive index, extend current range
            current_end = indices[i]
        else:
            # Gap found, close current range and start new one
            ranges.append((current_start, current_end + 1))
            current_start = indices[i]
            current_end = indices[i]
    
    # Add the final range
    ranges.append((current_start, current_end + 1))
    
    return ranges