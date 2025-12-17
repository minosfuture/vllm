# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Benchmark FlashInfer MoE backends to understand scaling with token batch size
and number of sequences in prefill-only and decode-only cases.

Focuses on NVFP4 quantization for DeepSeek-R1 model configuration.
Extensible to support additional backends and quantization types.
"""

import csv
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass

import torch

from vllm import _custom_ops as ops
from vllm.config import (
    ParallelConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.model_executor.layers.fused_moe.config import (
    fp8_w8a8_moe_quant_config,
    nvfp4_moe_quant_config,
)
from vllm.model_executor.layers.fused_moe.cutlass_moe import cutlass_moe_fp4
from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts, fused_topk
from vllm.model_executor.layers.quantization.utils.flashinfer_fp4_moe import (
    prepare_static_weights_for_trtllm_fp4_moe,
)
from vllm.platforms import current_platform
from vllm.scalar_type import scalar_types
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.utils.flashinfer import flashinfer_cutlass_fused_moe
from vllm.v1.worker.workspace import init_workspace_manager

# Suppress vLLM scheduler logs during benchmark (after imports)
logging.getLogger("vllm.config.scheduler").setLevel(logging.WARNING)

# Constants
FLOAT4_E2M1_MAX = scalar_types.float4_e2m1f.max()
FLOAT8_E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max

# Create a shared VllmConfig for benchmarking (created once to avoid log spam)
_BENCHMARK_VLLM_CONFIG = VllmConfig(
    parallel_config=ParallelConfig(pipeline_parallel_size=1),
)


@dataclass
class MoEConfig:
    """Configuration for MoE benchmarks (DeepSeek-R1 defaults)."""

    num_experts: int = 256
    topk: int = 8
    hidden_size: int = 7168
    intermediate_size: int = 2048
    dtype: torch.dtype = torch.bfloat16
    quant_dtype: str = "nvfp4"  # Extensible: "nvfp4", "fp8", None


@dataclass
class BenchmarkResult:
    """Result from a single benchmark run."""

    backend: str
    num_seqs: int
    tokens_per_seq: int
    total_tokens: int
    num_experts: int
    topk: int
    hidden_size: int
    intermediate_size: int
    quant_dtype: str
    latency_us: float
    throughput_tokens_s: float
    phase: str  # "decode" or "prefill"


# Registry for benchmark backends
BACKEND_REGISTRY: dict[str, Callable] = {}


def register_backend(name: str):
    """Decorator to register benchmark backends."""

    def decorator(fn: Callable) -> Callable:
        BACKEND_REGISTRY[name] = fn
        return fn

    return decorator


def create_nvfp4_weights(
    num_experts: int,
    hidden_size: int,
    intermediate_size: int,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> dict:
    """Create NVFP4 quantized weights for MoE benchmark."""
    # Generate random weights
    w1 = (
        torch.randn(
            (num_experts, 2 * intermediate_size, hidden_size),
            device=device,
            dtype=dtype,
        )
        / 10
    )
    w2 = (
        torch.randn(
            (num_experts, hidden_size, intermediate_size), device=device, dtype=dtype
        )
        / 10
    )

    quant_blocksize = 16

    # Allocate quantized weight tensors
    w1_fp4 = torch.empty(
        (num_experts, 2 * intermediate_size, hidden_size // 2),
        device=device,
        dtype=torch.uint8,
    )
    w2_fp4 = torch.empty(
        (num_experts, hidden_size, intermediate_size // 2),
        device=device,
        dtype=torch.uint8,
    )

    w1_blockscale = torch.empty(
        (num_experts, 2 * intermediate_size, hidden_size // quant_blocksize),
        device=device,
        dtype=torch.float8_e4m3fn,
    )
    w2_blockscale = torch.empty(
        (num_experts, hidden_size, intermediate_size // quant_blocksize),
        device=device,
        dtype=torch.float8_e4m3fn,
    )

    w1_gs = torch.empty((num_experts,), device=device, dtype=torch.float32)
    w2_gs = torch.empty((num_experts,), device=device, dtype=torch.float32)
    a1_gs = torch.ones((num_experts,), device=device, dtype=torch.float32)
    a2_gs = torch.ones((num_experts,), device=device, dtype=torch.float32)

    # Quantize weights per expert
    for expert in range(num_experts):
        w1_e = w1[expert]
        w2_e = w2[expert]
        w1_amax = torch.abs(w1_e).max().to(torch.float32)
        w2_amax = torch.abs(w2_e).max().to(torch.float32)
        w1_gs[expert] = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / w1_amax
        w2_gs[expert] = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / w2_amax

        w1_fp4[expert], w1_blockscale[expert] = ops.scaled_fp4_quant(
            w1_e, w1_gs[expert]
        )
        w2_fp4[expert], w2_blockscale[expert] = ops.scaled_fp4_quant(
            w2_e, w2_gs[expert]
        )

    return {
        "w1": w1,
        "w2": w2,
        "w1_fp4": w1_fp4,
        "w2_fp4": w2_fp4,
        "w1_blockscale": w1_blockscale,
        "w2_blockscale": w2_blockscale,
        "w1_gs": w1_gs,
        "w2_gs": w2_gs,
        "a1_gs": a1_gs,
        "a2_gs": a2_gs,
    }


def create_fp8_weights(
    num_experts: int,
    hidden_size: int,
    intermediate_size: int,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> dict:
    """Create FP8 quantized weights for MoE benchmark."""
    w1 = (
        torch.randn(
            (num_experts, 2 * intermediate_size, hidden_size),
            device=device,
            dtype=dtype,
        )
        / 10
    )
    w2 = (
        torch.randn(
            (num_experts, hidden_size, intermediate_size), device=device, dtype=dtype
        )
        / 10
    )

    w1_fp8 = torch.empty(
        (num_experts, 2 * intermediate_size, hidden_size),
        device=device,
        dtype=torch.float8_e4m3fn,
    )
    w2_fp8 = torch.empty(
        (num_experts, hidden_size, intermediate_size),
        device=device,
        dtype=torch.float8_e4m3fn,
    )
    w1_scale = torch.empty((num_experts, 1, 1), device=device, dtype=torch.float32)
    w2_scale = torch.empty((num_experts, 1, 1), device=device, dtype=torch.float32)

    for expert in range(num_experts):
        w1_fp8[expert], w1_scale[expert] = ops.scaled_fp8_quant(w1[expert])
        w2_fp8[expert], w2_scale[expert] = ops.scaled_fp8_quant(w2[expert])

    return {
        "w1": w1,
        "w2": w2,
        "w1_fp8": w1_fp8,
        "w2_fp8": w2_fp8,
        "w1_scale": w1_scale,
        "w2_scale": w2_scale,
    }


def create_flashinfer_trtllm_nvfp4_weights(
    num_experts: int,
    hidden_size: int,
    intermediate_size: int,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> dict:
    """Create TRT-LLM NVFP4 weights with proper shuffling for benchmark."""
    import flashinfer

    # Create base weights in the shape expected by TRT-LLM kernel
    # w1 (gate/up): [num_experts, 2 * intermediate_size, hidden_size]
    # w2 (down): [num_experts, hidden_size, intermediate_size]
    w1 = (
        torch.randn(
            (num_experts, 2 * intermediate_size, hidden_size),
            device=device,
            dtype=dtype,
        )
        / 10
    )
    w2 = (
        torch.randn(
            (num_experts, hidden_size, intermediate_size), device=device, dtype=dtype
        )
        / 10
    )

    # Compute global scale for input quantization
    w1_amax = torch.abs(w1).max().to(torch.float32)
    a1_gscale = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / w1_amax

    # Quantize using flashinfer's fp4_quantize
    # First flatten all experts together for quantization
    w1_flat = w1.reshape(-1, hidden_size)
    w2_flat = w2.reshape(-1, intermediate_size)

    # Quantize to FP4
    w1_fp4_packed, w1_scales = flashinfer.fp4_quantize(
        w1_flat, a1_gscale, is_sf_swizzled_layout=False
    )
    w2_amax = torch.abs(w2).max().to(torch.float32)
    a2_gscale = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / w2_amax
    w2_fp4_packed, w2_scales = flashinfer.fp4_quantize(
        w2_flat, a2_gscale, is_sf_swizzled_layout=False
    )

    # Reshape back to per-expert
    w1_fp4_packed = w1_fp4_packed.reshape(num_experts, 2 * intermediate_size, -1)
    w1_scales = w1_scales.reshape(num_experts, 2 * intermediate_size, -1)
    w2_fp4_packed = w2_fp4_packed.reshape(num_experts, hidden_size, -1)
    w2_scales = w2_scales.reshape(num_experts, hidden_size, -1)

    # Prepare shuffled weights for TRT-LLM kernel
    (
        gemm1_weights_shuffled,
        gemm1_scales_shuffled,
        gemm2_weights_shuffled,
        gemm2_scales_shuffled,
    ) = prepare_static_weights_for_trtllm_fp4_moe(
        gemm1_weights=w1_fp4_packed,
        gemm2_weights=w2_fp4_packed,
        gemm1_scales_linear_fp4_bytes=w1_scales,
        gemm2_scales_linear_fp4_bytes=w2_scales,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=num_experts,
    )

    # Create output scales (g1_scale_c, g1_alphas, g2_alphas)
    g1_scale_c = torch.ones(1, device=device, dtype=torch.float32)
    g1_alphas = torch.ones(1, device=device, dtype=torch.float32)
    g2_alphas = torch.ones(1, device=device, dtype=torch.float32)

    return {
        "w1": w1,
        "w2": w2,
        "a1_gscale": a1_gscale,
        "gemm1_weights_shuffled": gemm1_weights_shuffled,
        "gemm1_scales_shuffled": gemm1_scales_shuffled,
        "gemm2_weights_shuffled": gemm2_weights_shuffled,
        "gemm2_scales_shuffled": gemm2_scales_shuffled,
        "g1_scale_c": g1_scale_c,
        "g1_alphas": g1_alphas,
        "g2_alphas": g2_alphas,
    }


@register_backend("cutlass_moe_fp4")
def benchmark_cutlass_moe_fp4(
    config: MoEConfig,
    num_tokens: int,
    weights: dict,
    num_warmup: int = 10,
    num_iters: int = 100,
) -> float:
    """Benchmark CUTLASS NVFP4 MoE kernel."""
    device = "cuda"
    a = torch.randn((num_tokens, config.hidden_size), device=device, dtype=config.dtype)
    score = torch.randn(
        (num_tokens, config.num_experts), device=device, dtype=config.dtype
    )

    topk_weights, topk_ids, _ = fused_topk(a, score, config.topk, renormalize=False)

    quant_config = nvfp4_moe_quant_config(
        a1_gscale=weights["a1_gs"],
        a2_gscale=weights["a2_gs"],
        w1_scale=weights["w1_blockscale"],
        w2_scale=weights["w2_blockscale"],
        g1_alphas=weights["w1_gs"],
        g2_alphas=weights["w2_gs"],
    )

    def run_kernel():
        with set_current_vllm_config(_BENCHMARK_VLLM_CONFIG):
            return cutlass_moe_fp4(
                a=a,
                w1_fp4=weights["w1_fp4"],
                w2_fp4=weights["w2_fp4"],
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                m=num_tokens,
                n=config.intermediate_size,
                k=config.hidden_size,
                e=config.num_experts,
                quant_config=quant_config,
            )

    # Warmup
    for _ in range(num_warmup):
        run_kernel()
    torch.cuda.synchronize()

    # Benchmark
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(num_iters):
        run_kernel()
    end_event.record()
    end_event.synchronize()

    latency_ms = start_event.elapsed_time(end_event) / num_iters
    return latency_ms * 1000  # Return in microseconds


@register_backend("flashinfer_cutlass_nvfp4")
def benchmark_flashinfer_cutlass_nvfp4(
    config: MoEConfig,
    num_tokens: int,
    weights: dict,
    num_warmup: int = 10,
    num_iters: int = 100,
) -> float:
    """Benchmark FlashInfer CUTLASS NVFP4 MoE kernel (direct kernel call)."""
    from flashinfer.fused_moe.core import ActivationType

    device = "cuda"
    a = torch.randn((num_tokens, config.hidden_size), device=device, dtype=config.dtype)
    score = torch.randn(
        (num_tokens, config.num_experts), device=device, dtype=config.dtype
    )

    topk_weights, topk_ids, _ = fused_topk(a, score, config.topk, renormalize=False)

    # Prepare quant_scales for NVFP4:
    # [a1_gscale, w1_blockscale, g1_alphas, a2_gscale, w2_blockscale, g2_alphas]
    quant_scales = [
        weights["a1_gs"],  # gemm1 activation global scale
        weights["w1_blockscale"].view(torch.int32),  # gemm1 weights block scales
        weights["w1_gs"],  # gemm1 dequant scale (g1_alphas)
        weights["a2_gs"],  # gemm2 activation global scale
        weights["w2_blockscale"].view(torch.int32),  # gemm2 weights block scales
        weights["w2_gs"],  # gemm2 dequant scale (g2_alphas)
    ]

    # FlashInfer API requires weight to be long for nvfp4
    fc1_weights = weights["w1_fp4"].view(torch.long)
    fc2_weights = weights["w2_fp4"].view(torch.long)

    # Pre-allocate output tensor
    output = torch.empty(
        (num_tokens, config.hidden_size), device=device, dtype=config.dtype
    )

    def run_kernel():
        return flashinfer_cutlass_fused_moe(
            input=a,
            token_selected_experts=topk_ids.to(torch.int),
            token_final_scales=topk_weights,
            fc1_expert_weights=fc1_weights,
            fc2_expert_weights=fc2_weights,
            output_dtype=config.dtype,
            quant_scales=quant_scales,
            output=output,
            activation_type=ActivationType.Swiglu,
        )

    # Warmup
    for _ in range(num_warmup):
        run_kernel()
    torch.cuda.synchronize()

    # Benchmark
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(num_iters):
        run_kernel()
    end_event.record()
    end_event.synchronize()

    latency_ms = start_event.elapsed_time(end_event) / num_iters
    return latency_ms * 1000  # Return in microseconds


@register_backend("triton_fp8")
def benchmark_triton_fp8(
    config: MoEConfig,
    num_tokens: int,
    weights: dict,
    num_warmup: int = 10,
    num_iters: int = 100,
) -> float:
    """Benchmark Triton FP8 MoE kernel (baseline)."""
    device = "cuda"
    a = torch.randn((num_tokens, config.hidden_size), device=device, dtype=config.dtype)
    score = torch.randn(
        (num_tokens, config.num_experts), device=device, dtype=config.dtype
    )

    _, a_fp8_scale = ops.scaled_fp8_quant(a)
    topk_weights, topk_ids, _ = fused_topk(a, score, config.topk, renormalize=False)

    quant_config = fp8_w8a8_moe_quant_config(
        w1_scale=weights["w1_scale"],
        w2_scale=weights["w2_scale"],
        a1_scale=a_fp8_scale,
    )

    def run_kernel():
        with set_current_vllm_config(_BENCHMARK_VLLM_CONFIG):
            return fused_experts(
                a,
                weights["w1_fp8"],
                weights["w2_fp8"],
                topk_weights,
                topk_ids,
                quant_config=quant_config,
            )

    # Warmup
    for _ in range(num_warmup):
        run_kernel()
    torch.cuda.synchronize()

    # Benchmark
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(num_iters):
        run_kernel()
    end_event.record()
    end_event.synchronize()

    latency_ms = start_event.elapsed_time(end_event) / num_iters
    return latency_ms * 1000  # Return in microseconds


@register_backend("flashinfer_trtllm_nvfp4")
def benchmark_flashinfer_trtllm_nvfp4(
    config: MoEConfig,
    num_tokens: int,
    weights: dict,
    num_warmup: int = 10,
    num_iters: int = 100,
) -> float:
    """Benchmark TRT-LLM NVFP4 MoE kernel."""
    import flashinfer

    device = "cuda"
    a = torch.randn((num_tokens, config.hidden_size), device=device, dtype=config.dtype)
    score = torch.randn(
        (num_tokens, config.num_experts), device=device, dtype=torch.float32
    )

    # Quantize input using flashinfer's fp4_quantize
    a1_gscale = weights["a1_gscale"]
    hidden_states_fp4, hidden_states_scale = flashinfer.fp4_quantize(
        a, a1_gscale, is_sf_swizzled_layout=False
    )

    def run_kernel():
        return flashinfer.fused_moe.trtllm_fp4_block_scale_moe(
            routing_logits=score,
            routing_bias=None,
            hidden_states=hidden_states_fp4,
            hidden_states_scale=hidden_states_scale.view(torch.float8_e4m3fn).flatten(),
            gemm1_weights=weights["gemm1_weights_shuffled"],
            gemm1_weights_scale=weights["gemm1_scales_shuffled"].view(
                torch.float8_e4m3fn
            ),
            gemm1_bias=None,
            gemm1_alpha=None,
            gemm1_beta=None,
            gemm1_clamp_limit=None,
            gemm2_weights=weights["gemm2_weights_shuffled"],
            gemm2_weights_scale=weights["gemm2_scales_shuffled"].view(
                torch.float8_e4m3fn
            ),
            gemm2_bias=None,
            output1_scale_scalar=weights["g1_scale_c"],
            output1_scale_gate_scalar=weights["g1_alphas"],
            output2_scale_scalar=weights["g2_alphas"],
            num_experts=config.num_experts,
            top_k=config.topk,
            n_group=0,
            topk_group=0,
            intermediate_size=config.intermediate_size,
            local_expert_offset=0,
            local_num_experts=config.num_experts,
            routed_scaling_factor=None,
            tile_tokens_dim=None,
            routing_method_type=1,  # DeepSeekV3
            do_finalize=True,
        )[0]

    # Warmup
    for _ in range(num_warmup):
        run_kernel()
    torch.cuda.synchronize()

    # Benchmark
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(num_iters):
        run_kernel()
    end_event.record()
    end_event.synchronize()

    latency_ms = start_event.elapsed_time(end_event) / num_iters
    return latency_ms * 1000  # Return in microseconds


def run_benchmark(
    backend: str,
    config: MoEConfig,
    num_seqs: int,
    tokens_per_seq: int,
    weights: dict,
    num_warmup: int = 10,
    num_iters: int = 100,
) -> BenchmarkResult:
    """Run a single benchmark configuration."""
    if backend not in BACKEND_REGISTRY:
        raise ValueError(
            f"Unknown backend: {backend}. Available: {list(BACKEND_REGISTRY.keys())}"
        )

    num_tokens = num_seqs * tokens_per_seq
    benchmark_fn = BACKEND_REGISTRY[backend]
    latency_us = benchmark_fn(config, num_tokens, weights, num_warmup, num_iters)

    throughput = num_tokens / (latency_us / 1e6)
    phase = "decode" if tokens_per_seq == 1 else "prefill"

    return BenchmarkResult(
        backend=backend,
        num_seqs=num_seqs,
        tokens_per_seq=tokens_per_seq,
        total_tokens=num_tokens,
        num_experts=config.num_experts,
        topk=config.topk,
        hidden_size=config.hidden_size,
        intermediate_size=config.intermediate_size,
        quant_dtype=config.quant_dtype,
        latency_us=latency_us,
        throughput_tokens_s=throughput,
        phase=phase,
    )


def print_results_table(results: list[BenchmarkResult], title: str):
    """Print results in a formatted table."""
    print("=" * 100)
    print(title)
    print("=" * 100)
    print(
        f"{'Backend':<20} | {'Phase':<8} | {'Num Seqs':>10} | {'Tok/Seq':>8} | "
        f"{'Total Tok':>10} | {'Latency (us)':>14} | {'Throughput (tok/s)':>18}"
    )
    print("-" * 100)

    for r in results:
        print(
            f"{r.backend:<20} | {r.phase:<8} | {r.num_seqs:>10} | "
            f"{r.tokens_per_seq:>8} | {r.total_tokens:>10} | "
            f"{r.latency_us:>14.2f} | {r.throughput_tokens_s:>18,.0f}"
        )
    print("=" * 100)


def save_results_csv(results: list[BenchmarkResult], filepath: str):
    """Save results to CSV file."""
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "backend",
                "phase",
                "num_seqs",
                "tokens_per_seq",
                "total_tokens",
                "num_experts",
                "topk",
                "hidden_size",
                "intermediate_size",
                "quant_dtype",
                "latency_us",
                "throughput_tokens_s",
            ]
        )
        for r in results:
            writer.writerow(
                [
                    r.backend,
                    r.phase,
                    r.num_seqs,
                    r.tokens_per_seq,
                    r.total_tokens,
                    r.num_experts,
                    r.topk,
                    r.hidden_size,
                    r.intermediate_size,
                    r.quant_dtype,
                    r.latency_us,
                    r.throughput_tokens_s,
                ]
            )
    print(f"Results saved to: {filepath}")


def save_results_json(results: list[BenchmarkResult], filepath: str):
    """Save results to JSON file."""
    data = {
        "benchmark_name": "flashinfer_moe",
        "model": "DeepSeek-R1",
        "device": torch.cuda.get_device_name(0),
        "results": [
            {
                "backend": r.backend,
                "phase": r.phase,
                "num_seqs": r.num_seqs,
                "tokens_per_seq": r.tokens_per_seq,
                "total_tokens": r.total_tokens,
                "num_experts": r.num_experts,
                "topk": r.topk,
                "hidden_size": r.hidden_size,
                "intermediate_size": r.intermediate_size,
                "quant_dtype": r.quant_dtype,
                "latency_us": r.latency_us,
                "throughput_tokens_s": r.throughput_tokens_s,
            }
            for r in results
        ],
    }
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Results saved to: {filepath}")


def generate_figures(
    results: list[BenchmarkResult],
    output_dir: str,
    device_name: str,
):
    """Generate scaling figures for each backend and phase."""
    import os

    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)

    # Group results by backend and phase
    from collections import defaultdict

    grouped: dict[tuple[str, str], list[BenchmarkResult]] = defaultdict(list)
    for r in results:
        grouped[(r.backend, r.phase)].append(r)

    # Color palette for different tokens_per_seq (seq_lens)
    colors = plt.cm.viridis([0.2, 0.5, 0.8, 0.95])

    for (backend, phase), phase_results in grouped.items():
        if not phase_results:
            continue

        # Create figure with two subplots: latency and throughput
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        if phase == "decode":
            # For decode: x-axis is num_seqs (tokens_per_seq = 1)
            data = sorted(phase_results, key=lambda x: x.num_seqs)

            num_seqs = [r.num_seqs for r in data]
            latencies = [r.latency_us for r in data]
            throughputs = [r.throughput_tokens_s / 1000 for r in data]  # K tok/s

            axes[0].plot(num_seqs, latencies, "o-", color=colors[0], linewidth=2)
            axes[1].plot(num_seqs, throughputs, "o-", color=colors[0], linewidth=2)

            # Add data labels
            for x, y in zip(num_seqs, latencies):
                axes[0].annotate(
                    f"{y:.0f}",
                    (x, y),
                    textcoords="offset points",
                    xytext=(0, 8),
                    ha="center",
                    fontsize=8,
                )
            for x, y in zip(num_seqs, throughputs):
                axes[1].annotate(
                    f"{y:.1f}",
                    (x, y),
                    textcoords="offset points",
                    xytext=(0, 8),
                    ha="center",
                    fontsize=8,
                )

            axes[0].set_xlabel("Number of Sequences (Batch Size)", fontsize=12)
            axes[1].set_xlabel("Number of Sequences (Batch Size)", fontsize=12)

        else:  # prefill
            # For prefill: x-axis is total_tokens, different lines for tokens_per_seq
            tokens_per_seqs = sorted(set(r.tokens_per_seq for r in phase_results))

            for idx, tps in enumerate(tokens_per_seqs):
                data = [r for r in phase_results if r.tokens_per_seq == tps]
                data.sort(key=lambda x: x.total_tokens)

                total_tokens = [r.total_tokens for r in data]
                latencies = [r.latency_us for r in data]
                throughputs = [r.throughput_tokens_s / 1e6 for r in data]  # M tok/s

                color = colors[idx % len(colors)]
                label = f"seq_len={tps}"

                axes[0].plot(
                    total_tokens, latencies, "o-", color=color, label=label, linewidth=2
                )
                axes[1].plot(
                    total_tokens,
                    throughputs,
                    "o-",
                    color=color,
                    label=label,
                    linewidth=2,
                )

                # Add data labels
                offset_y = 8 if idx % 2 == 0 else -12
                for x, y in zip(total_tokens, latencies):
                    axes[0].annotate(
                        f"{y:.0f}",
                        (x, y),
                        textcoords="offset points",
                        xytext=(0, offset_y),
                        ha="center",
                        fontsize=7,
                        color=color,
                    )
                for x, y in zip(total_tokens, throughputs):
                    axes[1].annotate(
                        f"{y:.2f}",
                        (x, y),
                        textcoords="offset points",
                        xytext=(0, offset_y),
                        ha="center",
                        fontsize=7,
                        color=color,
                    )

            axes[0].set_xlabel("Total Tokens", fontsize=12)
            axes[1].set_xlabel("Total Tokens", fontsize=12)
            axes[0].legend(loc="best", fontsize=10)
            axes[1].legend(loc="best", fontsize=10)

        # Configure latency subplot
        axes[0].set_ylabel("Latency (μs)", fontsize=12)
        axes[0].set_title(f"{backend} - Latency vs Batch Size", fontsize=14)
        axes[0].set_xscale("log", base=2)
        axes[0].set_ylim(bottom=0)
        axes[0].grid(True, alpha=0.3)

        # Configure throughput subplot
        throughput_unit = "K tok/s" if phase == "decode" else "M tok/s"
        axes[1].set_ylabel(f"Throughput ({throughput_unit})", fontsize=12)
        axes[1].set_title(f"{backend} - Throughput vs Batch Size", fontsize=14)
        axes[1].set_xscale("log", base=2)
        axes[1].set_ylim(bottom=0)
        axes[1].grid(True, alpha=0.3)

        # Add overall title
        fig.suptitle(
            f"MoE {phase.capitalize()} Benchmark - {backend}\n"
            f"Device: {device_name} | Model: DeepSeek-R1 (256 experts, top-8)",
            fontsize=14,
            fontweight="bold",
        )
        plt.tight_layout()

        # Save figure
        filename = f"moe_{backend}_{phase}.png"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Figure saved to: {filepath}")

    # Generate combined comparison figure if multiple backends
    backends_by_phase: dict[str, list[str]] = defaultdict(list)
    for backend, phase in grouped:
        backends_by_phase[phase].append(backend)

    for phase, backends in backends_by_phase.items():
        if len(backends) < 2:
            continue

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        for idx, backend in enumerate(sorted(backends)):
            phase_results = grouped[(backend, phase)]
            if not phase_results:
                continue

            if phase == "decode":
                data = sorted(phase_results, key=lambda x: x.num_seqs)
                x_values = [r.num_seqs for r in data]
                throughputs = [r.throughput_tokens_s / 1000 for r in data]
            else:
                # Use the first tokens_per_seq for comparison
                tokens_per_seqs = sorted(set(r.tokens_per_seq for r in phase_results))
                if not tokens_per_seqs:
                    continue
                target_tps = tokens_per_seqs[0]

                data = [r for r in phase_results if r.tokens_per_seq == target_tps]
                data.sort(key=lambda x: x.total_tokens)
                x_values = [r.total_tokens for r in data]
                throughputs = [r.throughput_tokens_s / 1e6 for r in data]

            latencies = [r.latency_us for r in data]

            color = colors[idx % len(colors)]
            axes[0].plot(
                x_values, latencies, "o-", color=color, label=backend, linewidth=2
            )
            axes[1].plot(
                x_values, throughputs, "o-", color=color, label=backend, linewidth=2
            )

            # Add data labels
            offset_y = 8 if idx % 2 == 0 else -12
            for x, y in zip(x_values, latencies):
                axes[0].annotate(
                    f"{y:.0f}",
                    (x, y),
                    textcoords="offset points",
                    xytext=(0, offset_y),
                    ha="center",
                    fontsize=7,
                    color=color,
                )
            for x, y in zip(x_values, throughputs):
                axes[1].annotate(
                    f"{y:.1f}",
                    (x, y),
                    textcoords="offset points",
                    xytext=(0, offset_y),
                    ha="center",
                    fontsize=7,
                    color=color,
                )

        x_label = "Number of Sequences" if phase == "decode" else "Total Tokens"
        axes[0].set_xlabel(x_label, fontsize=12)
        axes[0].set_ylabel("Latency (μs)", fontsize=12)
        axes[0].set_title(f"Backend Comparison - Latency ({phase})", fontsize=14)
        axes[0].set_xscale("log", base=2)
        axes[0].set_ylim(bottom=0)
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(loc="best", fontsize=10)

        throughput_unit = "K tok/s" if phase == "decode" else "M tok/s"
        axes[1].set_xlabel(x_label, fontsize=12)
        axes[1].set_ylabel(f"Throughput ({throughput_unit})", fontsize=12)
        axes[1].set_title(f"Backend Comparison - Throughput ({phase})", fontsize=14)
        axes[1].set_xscale("log", base=2)
        axes[1].set_ylim(bottom=0)
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(loc="best", fontsize=10)

        extra_info = ""
        if phase == "prefill":
            extra_info = f" | seq_len={target_tps}"
        fig.suptitle(
            f"MoE {phase.capitalize()} Backend Comparison\n"
            f"Device: {device_name}{extra_info}",
            fontsize=14,
            fontweight="bold",
        )
        plt.tight_layout()

        filename = f"moe_comparison_{phase}.png"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Figure saved to: {filepath}")


def main():
    parser = FlexibleArgumentParser(
        description="Benchmark FlashInfer MoE backends (NVFP4 focus)"
    )
    parser.add_argument(
        "--backends",
        nargs="+",
        type=str,
        default=["cutlass_nvfp4"],
        choices=list(BACKEND_REGISTRY.keys()),
        help="Backends to benchmark",
    )
    parser.add_argument(
        "--decode-num-seqs",
        nargs="+",
        type=int,
        default=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048],
        help="Number of sequences for decode benchmarks",
    )
    parser.add_argument(
        "--prefill-num-seqs",
        nargs="+",
        type=int,
        default=[1, 2, 4, 8, 16, 32, 64],
        help="Number of sequences for prefill benchmarks",
    )
    parser.add_argument(
        "--prefill-seq-lens",
        nargs="+",
        type=int,
        default=[512, 1024, 2048],
        help="Sequence lengths for prefill benchmarks",
    )
    parser.add_argument(
        "--num-experts",
        type=int,
        default=256,
        help="Number of experts (DeepSeek-R1: 256)",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=8,
        help="Top-k experts per token (DeepSeek-R1: 8)",
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=7168,
        help="Hidden size (DeepSeek-R1: 7168)",
    )
    parser.add_argument(
        "--intermediate-size",
        type=int,
        default=2048,
        help="Intermediate size (DeepSeek-R1: 2048)",
    )
    parser.add_argument(
        "--num-warmup",
        type=int,
        default=10,
        help="Number of warmup iterations",
    )
    parser.add_argument(
        "--num-iters",
        type=int,
        default=100,
        help="Number of benchmark iterations",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help="Output CSV file path",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Output JSON file path",
    )
    parser.add_argument(
        "--output-figures",
        type=str,
        default=None,
        help="Output directory for figures (generates PNG files)",
    )
    parser.add_argument(
        "--decode-only",
        action="store_true",
        help="Run only decode benchmarks",
    )
    parser.add_argument(
        "--prefill-only",
        action="store_true",
        help="Run only prefill benchmarks",
    )

    args = parser.parse_args()

    # Check matplotlib availability if figures are requested
    if args.output_figures:
        try:
            import matplotlib.pyplot as plt  # noqa: F401
        except ImportError:
            raise ImportError(
                "matplotlib is required for figure generation. "
                "Install it with: pip install matplotlib"
            ) from None

    # Check device capability
    if not current_platform.has_device_capability(100):
        print(
            "WARNING: FlashInfer NVFP4 MoE requires SM100 (Blackwell). "
            "Some backends may not work."
        )

    # Create config
    config = MoEConfig(
        num_experts=args.num_experts,
        topk=args.topk,
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
    )

    print(f"MoE Configuration: {config}")
    print(f"Backends: {args.backends}")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print()

    # Create weights (once, reused across benchmarks)
    print("Creating NVFP4 weights (CUTLASS)...")
    nvfp4_weights = create_nvfp4_weights(
        config.num_experts,
        config.hidden_size,
        config.intermediate_size,
    )
    print("Creating FP8 weights...")
    fp8_weights = create_fp8_weights(
        config.num_experts,
        config.hidden_size,
        config.intermediate_size,
    )

    # Create TRT-LLM NVFP4 weights if needed
    trtllm_weights = None
    if "flashinfer_trtllm_nvfp4" in args.backends:
        print("Creating TRT-LLM NVFP4 weights (with shuffling)...")
        trtllm_weights = create_flashinfer_trtllm_nvfp4_weights(
            config.num_experts,
            config.hidden_size,
            config.intermediate_size,
        )
    print("Weights created.\n")

    def get_weights_for_backend(backend: str) -> dict:
        """Select appropriate weights for the given backend."""
        if backend == "flashinfer_trtllm_nvfp4":
            return trtllm_weights
        elif "nvfp4" in backend:
            return nvfp4_weights
        else:
            return fp8_weights

    results: list[BenchmarkResult] = []

    # Run decode benchmarks (tokens_per_seq = 1)
    if not args.prefill_only:
        print("Running DECODE benchmarks...")
        for backend in args.backends:
            weights = get_weights_for_backend(backend)
            for num_seqs in args.decode_num_seqs:
                try:
                    result = run_benchmark(
                        backend=backend,
                        config=config,
                        num_seqs=num_seqs,
                        tokens_per_seq=1,
                        weights=weights,
                        num_warmup=args.num_warmup,
                        num_iters=args.num_iters,
                    )
                    results.append(result)
                    print(
                        f"  {backend} | num_seqs={num_seqs:>5} | "
                        f"latency={result.latency_us:>10.2f} us | "
                        f"throughput={result.throughput_tokens_s:>12,.0f} tok/s"
                    )
                except Exception as e:
                    print(f"  {backend} | num_seqs={num_seqs:>5} | ERROR: {e}")

    # Run prefill benchmarks (tokens_per_seq > 1)
    if not args.decode_only:
        print("\nRunning PREFILL benchmarks...")
        for backend in args.backends:
            weights = get_weights_for_backend(backend)
            for num_seqs in args.prefill_num_seqs:
                for seq_len in args.prefill_seq_lens:
                    total_tokens = num_seqs * seq_len
                    if total_tokens > 131072:
                        continue  # Skip if exceeds max
                    try:
                        result = run_benchmark(
                            backend=backend,
                            config=config,
                            num_seqs=num_seqs,
                            tokens_per_seq=seq_len,
                            weights=weights,
                            num_warmup=args.num_warmup,
                            num_iters=args.num_iters,
                        )
                        results.append(result)
                        print(
                            f"  {backend} | num_seqs={num_seqs:>3} x "
                            f"seq_len={seq_len:>5} = "
                            f"{total_tokens:>6} tokens | "
                            f"latency={result.latency_us:>10.2f} us | "
                            f"throughput={result.throughput_tokens_s:>12,.0f} tok/s"
                        )
                    except Exception as e:
                        print(
                            f"  {backend} | num_seqs={num_seqs:>3} x "
                            f"seq_len={seq_len:>5} | ERROR: {e}"
                        )

    # Print summary table
    if results:
        print("\n")
        print_results_table(
            results,
            f"FlashInfer MoE Benchmark Results - {torch.cuda.get_device_name(0)}",
        )

        # Save results
        if args.output_csv:
            save_results_csv(results, args.output_csv)
        if args.output_json:
            save_results_json(results, args.output_json)
        if args.output_figures:
            generate_figures(
                results, args.output_figures, torch.cuda.get_device_name(0)
            )


if __name__ == "__main__":
    torch.set_default_device("cuda")
    current_platform.seed_everything(42)
    # Initialize workspace manager (required for CUTLASS MoE kernels)
    init_workspace_manager(torch.device("cuda"))
    main()
