# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Test V2 offloading correctness with DeepSeek V2 model."""

from ..utils import compare_two_settings

# def test_v2_offload_deepseek():
#    """Test V2 CPU offloading with DeepSeek-V2-Lite.
#
#    Compares outputs between:
#    1. Baseline (no offloading)
#    2. V2 offloading (group_size=8, num_in_group=2, prefetch_step=1)
#
#    This tests the advanced offloading with prefetching on a MoE model.
#    """
#    compare_two_settings(
#        "deepseek-ai/DeepSeek-V2-Lite",
#        [],  # Baseline: no offloading
#        [
#            # V2 offloading configuration
#            "--offload-group-size",
#            "8",
#            "--offload-num-in-group",
#            "2",
#            "--offload-prefetch-step",
#            "1",
#        ],
#    )


def test_v2_offload_small_model():
    """Test V2 offloading with a smaller model for quick validation.

    Uses a tiny model to verify basic V2 offloading functionality works.
    Uses --kv-cache-memory-bytes to ensure same KV cache size for fair comparison.
    """
    # Use fixed KV cache size (1GB) for both baseline and V2 offload
    # This ensures we're comparing apples to apples - same KV cache,
    # but V2 saves GPU memory by offloading model weights
    kv_cache_bytes = str(1 * 1024 * 1024 * 1024)  # 1 GB
    common_args = [
        "--kv-cache-memory-bytes",
        kv_cache_bytes,
    ]
    compare_two_settings(
        "hmellor/tiny-random-LlamaForCausalLM",
        common_args,  # Baseline with fixed KV cache
        common_args
        + [
            # V2 offloading: offload 1 out of every 4 layers
            "--offload-group-size",
            "4",
            "--offload-num-in-group",
            "1",
            "--offload-prefetch-step",
            "1",
        ],
    )
