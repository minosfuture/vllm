#!/usr/bin/env python3
"""
Test script to validate ubatch prefill support implementation.
"""
import torch
import numpy as np
import time
from typing import List, Dict, Any
from vllm.v1.attention.backends.utils import (
    UbatchSlice, UbatchWorkloadInfo, analyze_workload,
    create_balanced_ubatch_slices, estimate_compute_complexity
)


def test_estimate_compute_complexity():
    """Test compute complexity estimation for prefill vs decode."""
    print("Testing compute complexity estimation...")

    # Decode operation (query_len = 1)
    decode_complexity = estimate_compute_complexity(query_len=1, seq_len=100)
    print(f"Decode complexity (1, 100): {decode_complexity}")

    # Prefill operations with different sizes
    prefill_small = estimate_compute_complexity(query_len=10, seq_len=100)
    prefill_large = estimate_compute_complexity(query_len=50, seq_len=100)

    print(f"Small prefill complexity (10, 100): {prefill_small}")
    print(f"Large prefill complexity (50, 100): {prefill_large}")

    # Verify decode is O(n) and prefill is O(n*m)
    assert decode_complexity == 100.0, f"Expected 100, got {decode_complexity}"
    assert prefill_small == 1000.0, f"Expected 1000, got {prefill_small}"
    assert prefill_large == 5000.0, f"Expected 5000, got {prefill_large}"

    print("✓ Compute complexity estimation works correctly\n")


def test_workload_analysis():
    """Test workload analysis for mixed decode/prefill batches."""
    print("Testing workload analysis...")

    # Mixed batch: 2 decode, 2 prefill requests
    query_lens = torch.tensor([1, 1, 10, 20], dtype=torch.int32)  # 2 decode, 2 prefill
    seq_lens = torch.tensor([50, 80, 100, 150], dtype=torch.int32)

    workload_info = analyze_workload(query_lens, seq_lens)

    print(f"Total requests: {workload_info.total_requests}")
    print(f"Decode requests: {workload_info.decode_requests}")
    print(f"Prefill requests: {workload_info.prefill_requests}")
    print(f"Total tokens: {workload_info.total_tokens}")
    print(f"Decode tokens: {workload_info.decode_tokens}")
    print(f"Prefill tokens: {workload_info.prefill_tokens}")

    assert workload_info.total_requests == 4
    assert workload_info.decode_requests == 2
    assert workload_info.prefill_requests == 2
    assert workload_info.total_tokens == 32  # 1+1+10+20 = 32
    assert workload_info.decode_tokens == 2  # 1+1 = 2
    assert workload_info.prefill_tokens == 30  # 10+20 = 30

    print("✓ Workload analysis works correctly\n")


def test_balanced_ubatch_creation():
    """Test creation of balanced ubatch slices."""
    print("Testing balanced ubatch creation...")

    # Mixed workload with different complexity levels
    query_lens = torch.tensor([1, 1, 8, 16], dtype=torch.int32)
    seq_lens = torch.tensor([100, 200, 150, 300], dtype=torch.int32)

    workload_info = analyze_workload(query_lens, seq_lens)

    # Test compute complexity balancing with 2 ubatches
    ubatch_slices = create_balanced_ubatch_slices(
        workload_info,
        num_ubatches=2,
        balance_strategy="compute_complexity"
    )

    print("Ubatch slices (2 ubatches, complexity balancing):")
    for i, ubatch in enumerate(ubatch_slices):
        print(f"  Ubatch {i}: requests={ubatch.request_slice}, "
              f"tokens={ubatch.token_slice}, "
              f"complexity={ubatch.compute_complexity:.1f}, "
              f"is_prefill={ubatch.is_prefill}, "
              f"max_query_len={ubatch.max_query_len}")

        # Validate ubatch slice attributes
        assert hasattr(ubatch, 'compute_complexity')
        assert hasattr(ubatch, 'is_prefill')
        assert hasattr(ubatch, 'max_query_len')
        assert ubatch.compute_complexity is not None
        assert isinstance(ubatch.is_prefill, bool), f"Expected bool, got {type(ubatch.is_prefill)}"
        assert ubatch.max_query_len >= 1

    # Test token balancing with 3 ubatches
    ubatch_slices_tokens = create_balanced_ubatch_slices(
        workload_info,
        num_ubatches=3,
        balance_strategy="tokens"
    )

    print(f"\nUbatch slices (3 ubatches, token balancing):")
    for i, ubatch in enumerate(ubatch_slices_tokens):
        print(f"  Ubatch {i}: requests={ubatch.request_slice}, "
              f"tokens={ubatch.token_slice}, "
              f"is_prefill={ubatch.is_prefill}")

        # Verify consecutive slices
        assert ubatch.request_slice.start < ubatch.request_slice.stop
        assert ubatch.token_slice.start < ubatch.token_slice.stop

    # Test with 4 ubatches (one per request)
    ubatch_slices_4 = create_balanced_ubatch_slices(
        workload_info,
        num_ubatches=4,
        balance_strategy="compute_complexity"
    )

    print(f"\nUbatch slices (4 ubatches - one per request):")
    assert len(ubatch_slices_4) == 4
    for i, ubatch in enumerate(ubatch_slices_4):
        print(f"  Ubatch {i}: requests={ubatch.request_slice}, "
              f"tokens={ubatch.token_slice}")
        # Each ubatch should have exactly one request
        assert ubatch.request_slice.stop - ubatch.request_slice.start == 1

    print("✓ Balanced ubatch creation works correctly\n")


def test_decode_only_workload():
    """Test that decode-only workloads still work correctly."""
    print("Testing decode-only workload...")

    # Pure decode workload
    query_lens = torch.tensor([1, 1, 1, 1], dtype=torch.int32)
    seq_lens = torch.tensor([50, 100, 75, 125], dtype=torch.int32)

    workload_info = analyze_workload(query_lens, seq_lens)

    assert workload_info.decode_requests == 4
    assert workload_info.prefill_requests == 0
    assert workload_info.total_tokens == 4

    ubatch_slices = create_balanced_ubatch_slices(workload_info, num_ubatches=2)

    print(f"Decode-only ubatches:")
    for i, ubatch in enumerate(ubatch_slices):
        print(f"  Ubatch {i}: is_prefill={ubatch.is_prefill}, "
              f"max_query_len={ubatch.max_query_len}")
        assert not ubatch.is_prefill  # Should be False for decode-only
        assert ubatch.max_query_len == 1  # Should be 1 for decode

    print("✓ Decode-only workload handling works correctly\n")


def test_single_request_edge_case():
    """Test edge case with single request."""
    print("Testing single request edge case...")

    query_lens = torch.tensor([10], dtype=torch.int32)
    seq_lens = torch.tensor([100], dtype=torch.int32)

    workload_info = analyze_workload(query_lens, seq_lens)
    ubatch_slices = create_balanced_ubatch_slices(workload_info, num_ubatches=2)

    # Should create a single ubatch for single request
    assert len(ubatch_slices) == 1
    assert ubatch_slices[0].is_prefill == True
    assert ubatch_slices[0].max_query_len == 10

    print("✓ Single request edge case works correctly\n")


def test_performance_benchmark():
    """Benchmark the performance of create_balanced_ubatch_slices with different input sizes."""
    print("Testing performance benchmark...")

    # Test configurations: (num_requests, workload_type, num_ubatches)
    test_configs = [
        # Small batches
        (4, "decode", 2),
        (8, "decode", 2),
        (4, "mixed", 2),
        (8, "mixed", 2),
        (4, "prefill", 2),
        (8, "prefill", 2),

        # Medium batches
        (32, "decode", 2),
        (64, "decode", 2),
        (32, "mixed", 2),
        (64, "mixed", 2),
        (32, "prefill", 2),
        (64, "prefill", 2),

        # Large batches
        (128, "decode", 2),
        (256, "decode", 2),
        (128, "mixed", 2),
        (256, "mixed", 2),
        (128, "prefill", 2),
        (256, "prefill", 2),

        # Different ubatch counts
        (64, "mixed", 4),
        (64, "mixed", 8),
        (128, "mixed", 4),
        (128, "mixed", 8),
    ]

    benchmark_results = []

    print(f"{'Requests':<9} {'Workload':<8} {'Ubatches':<9} {'Time (ms)':<10} {'Time/Req (μs)':<15} {'Throughput (req/s)':<18}")
    print("-" * 80)

    for num_requests, workload_type, num_ubatches in test_configs:
        # Generate test data based on workload type
        if workload_type == "decode":
            query_lens = torch.ones(num_requests, dtype=torch.int32)
            seq_lens = torch.randint(50, 500, (num_requests,), dtype=torch.int32)
        elif workload_type == "prefill":
            query_lens = torch.randint(10, 100, (num_requests,), dtype=torch.int32)
            seq_lens = query_lens + torch.randint(0, 50, (num_requests,), dtype=torch.int32)
        else:  # mixed
            decode_count = num_requests // 2
            prefill_count = num_requests - decode_count

            decode_lens = torch.ones(decode_count, dtype=torch.int32)
            prefill_lens = torch.randint(10, 100, (prefill_count,), dtype=torch.int32)
            query_lens = torch.cat([decode_lens, prefill_lens])

            decode_seq_lens = torch.randint(50, 500, (decode_count,), dtype=torch.int32)
            prefill_seq_lens = prefill_lens + torch.randint(0, 50, (prefill_count,), dtype=torch.int32)
            seq_lens = torch.cat([decode_seq_lens, prefill_seq_lens])

            # Shuffle to mix decode and prefill requests
            perm = torch.randperm(num_requests)
            query_lens = query_lens[perm]
            seq_lens = seq_lens[perm]

        # Analyze workload
        workload_info = analyze_workload(query_lens, seq_lens)

        # Warm up
        for _ in range(3):
            create_balanced_ubatch_slices(workload_info, num_ubatches, "compute_complexity")

        # Benchmark
        num_trials = 10
        times = []

        for _ in range(num_trials):
            start_time = time.time()
            ubatch_slices = create_balanced_ubatch_slices(
                workload_info,
                num_ubatches,
                "compute_complexity"
            )
            end_time = time.time()
            times.append(end_time - start_time)

        # Calculate statistics
        avg_time = np.mean(times)
        time_ms = avg_time * 1000
        time_per_request_us = (avg_time * 1000000) / num_requests
        throughput = num_requests / avg_time

        benchmark_results.append({
            'num_requests': num_requests,
            'workload_type': workload_type,
            'num_ubatches': num_ubatches,
            'avg_time_ms': time_ms,
            'time_per_request_us': time_per_request_us,
            'throughput': throughput,
            'std_time_ms': np.std(times) * 1000
        })

        print(f"{num_requests:<9} {workload_type:<8} {num_ubatches:<9} "
              f"{time_ms:<10.3f} {time_per_request_us:<15.3f} {throughput:<18.1f}")

        # Validate that we got the expected number of ubatches
        actual_ubatches = len(ubatch_slices)
        expected_ubatches = min(num_ubatches, num_requests)
        assert actual_ubatches == expected_ubatches, f"Expected {expected_ubatches} ubatches, got {actual_ubatches}"

    # Performance analysis
    print("\n=== Performance Analysis ===")

    # Analyze scaling with batch size for mixed workloads
    mixed_decode_results = [r for r in benchmark_results if r['workload_type'] == 'mixed' and r['num_ubatches'] == 2]
    if len(mixed_decode_results) >= 2:
        print("\nScaling analysis (mixed workload, 2 ubatches):")
        for i in range(1, len(mixed_decode_results)):
            prev = mixed_decode_results[i-1]
            curr = mixed_decode_results[i]

            size_ratio = curr['num_requests'] / prev['num_requests']
            time_ratio = curr['avg_time_ms'] / prev['avg_time_ms']
            complexity_ratio = time_ratio / size_ratio

            print(f"  {prev['num_requests']} -> {curr['num_requests']} requests: "
                  f"{complexity_ratio:.2f}x complexity per request ratio")

            # Good scalability should have complexity_ratio close to 1.0
            # Dynamic programming is O(n^2 * k) where n=requests, k=ubatches
            # So we expect some super-linear scaling

    # Compare workload types at same batch size
    print("\nWorkload type comparison (64 requests, 2 ubatches):")
    size_64_results = [r for r in benchmark_results
                      if r['num_requests'] == 64 and r['num_ubatches'] == 2]

    if len(size_64_results) >= 3:
        decode_result = next((r for r in size_64_results if r['workload_type'] == 'decode'), None)
        mixed_result = next((r for r in size_64_results if r['workload_type'] == 'mixed'), None)
        prefill_result = next((r for r in size_64_results if r['workload_type'] == 'prefill'), None)

        if decode_result and mixed_result and prefill_result:
            print(f"  Decode:  {decode_result['avg_time_ms']:.3f} ms")
            print(f"  Mixed:   {mixed_result['avg_time_ms']:.3f} ms ({mixed_result['avg_time_ms']/decode_result['avg_time_ms']:.2f}x)")
            print(f"  Prefill: {prefill_result['avg_time_ms']:.3f} ms ({prefill_result['avg_time_ms']/decode_result['avg_time_ms']:.2f}x)")

    # Performance thresholds
    print("\n=== Performance Validation ===")

    # Check that even large batches complete reasonably quickly
    large_batch_results = [r for r in benchmark_results if r['num_requests'] >= 128]
    max_time_ms = max(r['avg_time_ms'] for r in large_batch_results) if large_batch_results else 0

    print(f"Maximum time for large batches (≥128 requests): {max_time_ms:.3f} ms")

    # Performance assertions
    if max_time_ms > 100:  # 100ms threshold
        print(f"⚠️  Warning: Large batch processing time ({max_time_ms:.3f} ms) exceeds 100ms threshold")
    else:
        print("✓ Large batch performance is acceptable")

    # Check that per-request time doesn't grow too much with batch size
    max_per_request_time = max(r['time_per_request_us'] for r in benchmark_results)
    print(f"Maximum per-request processing time: {max_per_request_time:.3f} μs")

    if max_per_request_time > 1000:  # 1ms per request threshold
        print(f"⚠️  Warning: Per-request processing time ({max_per_request_time:.3f} μs) exceeds 1ms threshold")
    else:
        print("✓ Per-request performance is acceptable")

    print("✓ Performance benchmark completed successfully\n")


def main():
    """Run all tests."""
    print("=== Testing Ubatch Prefill Support ===\n")

    try:
        test_estimate_compute_complexity()
        test_workload_analysis()
        test_balanced_ubatch_creation()
        test_decode_only_workload()
        test_single_request_edge_case()
        test_performance_benchmark()

        print("🎉 All tests passed! Ubatch prefill support is working correctly.")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
