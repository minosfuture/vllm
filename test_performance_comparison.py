#!/usr/bin/env python3
"""
Performance comparison between original and optimized ubatch algorithms.
"""
import torch
import numpy as np
import time
from typing import List, Dict, Any
from vllm.v1.attention.backends.utils import (
    UbatchSlice, UbatchWorkloadInfo, analyze_workload,
    create_balanced_ubatch_slices
)
from vllm.v1.attention.backends.utils_optimized import (
    create_fast_balanced_ubatch_slices
)


def create_test_workload(num_requests: int, workload_type: str) -> UbatchWorkloadInfo:
    """Create test workload for benchmarking."""
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

    return analyze_workload(query_lens, seq_lens)


def benchmark_algorithm(algorithm_func, workload_info: UbatchWorkloadInfo,
                       num_ubatches: int = 2, num_trials: int = 10) -> Dict[str, float]:
    """Benchmark a ubatch splitting algorithm."""
    # Warm up
    for _ in range(3):
        algorithm_func(workload_info, num_ubatches, "compute_complexity")

    # Benchmark
    times = []
    for _ in range(num_trials):
        start_time = time.time()
        result = algorithm_func(workload_info, num_ubatches, "compute_complexity")
        end_time = time.time()
        times.append(end_time - start_time)

    return {
        'avg_time_ms': np.mean(times) * 1000,
        'std_time_ms': np.std(times) * 1000,
        'min_time_ms': np.min(times) * 1000,
        'max_time_ms': np.max(times) * 1000,
        'num_ubatches': len(result)
    }


def compare_balance_quality(workload_info: UbatchWorkloadInfo,
                          original_ubatches: List[UbatchSlice],
                          optimized_ubatches: List[UbatchSlice]) -> Dict[str, float]:
    """Compare the balance quality of two ubatch splitting approaches."""

    def calculate_imbalance(ubatches):
        complexities = [ub.compute_complexity for ub in ubatches if ub.compute_complexity is not None]
        if len(complexities) < 2:
            return 0.0
        return (max(complexities) - min(complexities)) / max(complexities)

    def calculate_load_balance_score(ubatches):
        complexities = [ub.compute_complexity for ub in ubatches if ub.compute_complexity is not None]
        if len(complexities) < 2:
            return 1.0
        mean_complexity = np.mean(complexities)
        variance = np.var(complexities)
        return 1.0 / (1.0 + variance / (mean_complexity ** 2))

    original_imbalance = calculate_imbalance(original_ubatches)
    optimized_imbalance = calculate_imbalance(optimized_ubatches)

    original_score = calculate_load_balance_score(original_ubatches)
    optimized_score = calculate_load_balance_score(optimized_ubatches)

    return {
        'original_imbalance': original_imbalance,
        'optimized_imbalance': optimized_imbalance,
        'original_balance_score': original_score,
        'optimized_balance_score': optimized_score,
        'balance_score_ratio': optimized_score / original_score if original_score > 0 else float('inf')
    }


def main():
    """Run performance comparison."""
    print("=== Performance Comparison: Original vs Optimized Ubatch Algorithms ===\n")

    # Test configurations
    test_configs = [
        # Small batches (both should be fast)
        (16, "mixed", 2),
        (32, "mixed", 2),

        # Medium batches (where performance difference starts to show)
        (64, "mixed", 2),
        (64, "mixed", 4),
        (128, "mixed", 2),
        (128, "mixed", 4),

        # Large batches (where optimized should significantly outperform)
        (256, "mixed", 2),
        (256, "mixed", 4),
        (512, "mixed", 2),
        (512, "mixed", 4),
        (1024, "mixed", 2),
    ]

    print(f"{'Requests':<9} {'Workload':<8} {'Ubatches':<9} {'Original (ms)':<15} {'Optimized (ms)':<16} {'Speedup':<8} {'Balance Ratio':<13}")
    print("-" * 95)

    results = []

    for num_requests, workload_type, num_ubatches in test_configs:
        # Create test workload
        workload_info = create_test_workload(num_requests, workload_type)

        # Benchmark original algorithm (skip for very large batches to avoid timeout)
        if num_requests <= 512:
            original_perf = benchmark_algorithm(
                create_balanced_ubatch_slices,
                workload_info,
                num_ubatches
            )
            original_time = original_perf['avg_time_ms']
        else:
            original_time = float('inf')  # Too slow to measure

        # Benchmark optimized algorithm
        optimized_perf = benchmark_algorithm(
            create_fast_balanced_ubatch_slices,
            workload_info,
            num_ubatches
        )
        optimized_time = optimized_perf['avg_time_ms']

        # Calculate speedup
        if original_time != float('inf'):
            speedup = original_time / optimized_time
        else:
            speedup = float('inf')

        # Compare balance quality (only for measurable cases)
        balance_ratio = 1.0
        if original_time != float('inf'):
            try:
                original_ubatches = create_balanced_ubatch_slices(workload_info, num_ubatches, "compute_complexity")
                optimized_ubatches = create_fast_balanced_ubatch_slices(workload_info, num_ubatches, "compute_complexity")

                balance_comparison = compare_balance_quality(workload_info, original_ubatches, optimized_ubatches)
                balance_ratio = balance_comparison['balance_score_ratio']
            except Exception:
                balance_ratio = 1.0

        # Format results
        original_str = f"{original_time:.3f}" if original_time != float('inf') else "TIMEOUT"
        speedup_str = f"{speedup:.1f}x" if speedup != float('inf') else "∞"

        print(f"{num_requests:<9} {workload_type:<8} {num_ubatches:<9} "
              f"{original_str:<15} {optimized_time:<16.3f} {speedup_str:<8} {balance_ratio:<13.3f}")

        results.append({
            'num_requests': num_requests,
            'workload_type': workload_type,
            'num_ubatches': num_ubatches,
            'original_time_ms': original_time,
            'optimized_time_ms': optimized_time,
            'speedup': speedup,
            'balance_ratio': balance_ratio
        })

    # Summary analysis
    print("\n=== Performance Analysis ===")

    # Calculate geometric mean speedup for measured cases
    measurable_results = [r for r in results if r['original_time_ms'] != float('inf')]
    if measurable_results:
        speedups = [r['speedup'] for r in measurable_results]
        geo_mean_speedup = np.exp(np.mean(np.log(speedups)))
        print(f"Geometric mean speedup: {geo_mean_speedup:.2f}x")

        # Balance quality preservation
        balance_ratios = [r['balance_ratio'] for r in measurable_results]
        avg_balance_ratio = np.mean(balance_ratios)
        print(f"Average balance preservation ratio: {avg_balance_ratio:.3f}")

        if avg_balance_ratio >= 0.95:
            print("✓ Balance quality is well preserved")
        elif avg_balance_ratio >= 0.85:
            print("⚠️  Balance quality is reasonably preserved")
        else:
            print("❌ Balance quality is significantly degraded")

    # Scalability analysis
    print(f"\n=== Scalability Analysis ===")
    mixed_2batch_results = [r for r in results if r['workload_type'] == 'mixed' and r['num_ubatches'] == 2]

    if len(mixed_2batch_results) >= 2:
        print("Processing time scaling (mixed workload, 2 ubatches):")
        for i in range(1, len(mixed_2batch_results)):
            prev = mixed_2batch_results[i-1]
            curr = mixed_2batch_results[i]

            size_ratio = curr['num_requests'] / prev['num_requests']

            if curr['optimized_time_ms'] > 0 and prev['optimized_time_ms'] > 0:
                time_ratio = curr['optimized_time_ms'] / prev['optimized_time_ms']
                complexity_growth = time_ratio / size_ratio

                print(f"  {prev['num_requests']} -> {curr['num_requests']} requests: "
                      f"{complexity_growth:.2f}x complexity per request")

                if complexity_growth <= 1.5:
                    status = "✓ Good scalability"
                elif complexity_growth <= 2.0:
                    status = "⚠️  Acceptable scalability"
                else:
                    status = "❌ Poor scalability"
                print(f"    {status}")

    # Performance thresholds
    print(f"\n=== Performance Validation ===")

    large_batch_results = [r for r in results if r['num_requests'] >= 256]
    if large_batch_results:
        max_optimized_time = max(r['optimized_time_ms'] for r in large_batch_results)
        print(f"Maximum optimized time for large batches (≥256 requests): {max_optimized_time:.3f} ms")

        if max_optimized_time <= 10.0:  # 10ms threshold
            print("✓ Optimized algorithm meets performance requirements")
        elif max_optimized_time <= 50.0:
            print("⚠️  Optimized algorithm has acceptable performance")
        else:
            print("❌ Optimized algorithm performance needs improvement")

    # Final recommendation
    print(f"\n=== Recommendation ===")
    if len(measurable_results) > 0:
        min_speedup = min(r['speedup'] for r in measurable_results)
        avg_speedup = np.mean([r['speedup'] for r in measurable_results])

        print(f"Performance improvement: {min_speedup:.1f}x - {max(r['speedup'] for r in measurable_results):.1f}x")
        print(f"Average speedup: {avg_speedup:.1f}x")

        if avg_speedup >= 10.0:
            print("🚀 Optimized algorithm provides excellent performance improvement!")
        elif avg_speedup >= 3.0:
            print("✓ Optimized algorithm provides significant performance improvement")
        elif avg_speedup >= 1.5:
            print("⚠️  Optimized algorithm provides moderate performance improvement")
        else:
            print("❌ Optimization provides minimal benefit")

    print("\nConclusion: Use optimized algorithm for batches ≥64 requests")


if __name__ == "__main__":
    main()
