# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Benchmark FlashInfer MLA backends to understand scaling with token batch size
and number of sequences in prefill-only and decode-only cases.

Focuses on FP8 KV cache for DeepSeek-R1 model configuration.
Extensible to support additional backends and cache types.
"""

import csv
import json
from collections.abc import Callable
from dataclasses import dataclass

import torch

from vllm.platforms import current_platform
from vllm.utils.argparse_utils import FlexibleArgumentParser


@dataclass
class MLAConfig:
    """Configuration for MLA benchmarks (DeepSeek-R1 defaults)."""

    num_heads: int = 128
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    block_size: int = 64
    dtype: torch.dtype = torch.bfloat16
    kv_cache_dtype: str = "fp8_e4m3"  # Extensible: "fp8_e4m3", "auto"

    @property
    def qk_head_dim(self) -> int:
        """Total query-key head dimension."""
        return self.qk_nope_head_dim + self.qk_rope_head_dim

    @property
    def head_size(self) -> int:
        """KV cache head size = kv_lora_rank + qk_rope_head_dim."""
        return self.kv_lora_rank + self.qk_rope_head_dim


@dataclass
class BenchmarkResult:
    """Result from a single benchmark run."""

    backend: str
    num_seqs: int
    seq_len: int
    query_len: int
    total_tokens: int
    num_heads: int
    kv_lora_rank: int
    block_size: int
    kv_cache_dtype: str
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


def create_kv_cache(
    num_blocks: int,
    block_size: int,
    head_size: int,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    is_fp8: bool = True,
) -> torch.Tensor:
    """Create KV cache for MLA benchmark."""
    if is_fp8:
        # FP8 KV cache: pack with scale bytes
        # For FlashMLA sparse, the cache is stored as uint8 with extra scale bytes
        bytes_per_token = head_size + head_size // 64  # Add scale bytes
        kv_cache = (
            torch.randn(
                num_blocks,
                block_size,
                1,
                bytes_per_token,
                device=device,
                dtype=torch.float32,
            )
            .to(torch.float8_e4m3fn)
            .view(torch.uint8)
        )
    else:
        # BF16 KV cache
        kv_cache = torch.randn(
            num_blocks, block_size, head_size, device=device, dtype=dtype
        )
    return kv_cache


def create_block_tables(
    num_seqs: int,
    max_seq_len: int,
    block_size: int,
    device: str = "cuda",
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Create block tables and sequence lengths for benchmark."""
    # Random sequence lengths up to max_seq_len
    seq_lens = torch.randint(
        max_seq_len // 2, max_seq_len + 1, (num_seqs,), dtype=torch.int32, device=device
    )
    seq_lens[-1] = max_seq_len  # Ensure at least one has max length

    # Calculate blocks needed per sequence
    blocks_per_seq = (seq_lens + block_size - 1) // block_size
    max_blocks_per_seq = blocks_per_seq.max().item()
    total_blocks = blocks_per_seq.sum().item()

    # Create block tables with unique block IDs
    all_block_ids = torch.randperm(total_blocks, device=device)
    block_tables = torch.zeros(
        (num_seqs, max_blocks_per_seq), dtype=torch.int32, device=device
    )

    block_id = 0
    for i in range(num_seqs):
        num_blocks_needed = blocks_per_seq[i].item()
        block_tables[i, :num_blocks_needed] = all_block_ids[
            block_id : block_id + num_blocks_needed
        ]
        block_id += num_blocks_needed

    return block_tables, seq_lens, total_blocks


@register_backend("flashinfer_mla_decode")
def benchmark_flashinfer_mla_decode(
    config: MLAConfig,
    num_seqs: int,
    seq_len: int,
    query_len: int,
    num_warmup: int = 10,
    num_iters: int = 100,
) -> float:
    """Benchmark FlashInfer TRT-LLM MLA decode with FP8 KV cache."""
    from flashinfer.decode import trtllm_batch_decode_with_kv_cache_mla

    device = "cuda"

    # For FlashInfer MLA decode, query head_dim must match KV cache head_dim
    # qk_head_dim = kv_lora_rank + qk_rope_head_dim = 576 (same as head_size)
    # This is different from the logical qk_head_dim (qk_nope + qk_rope = 192)
    query_head_dim = config.head_size  # 576

    # Create query tensor: (batch_size, q_len_per_request, num_heads, head_size)
    q = torch.randn(
        num_seqs,
        query_len,
        config.num_heads,
        query_head_dim,
        device=device,
        dtype=config.dtype,
    )

    # Create block tables and sequence lengths
    block_tables, seq_lens, total_blocks = create_block_tables(
        num_seqs, seq_len, config.block_size, device
    )
    max_seq_len = seq_lens.max().item()

    # Create KV cache: (num_blocks, 1, block_size, head_size)
    # FlashInfer expects (num_blocks, num_kv_heads=1, block_size, head_dim)
    kv_cache = torch.randn(
        block_tables.numel(),
        config.block_size,
        config.head_size,
        device=device,
        dtype=config.dtype,
    ).unsqueeze(1)

    # Create workspace buffer
    workspace_buffer = torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device=device)

    # Compute scale
    scale = config.qk_head_dim**-0.5

    def run_kernel():
        return trtllm_batch_decode_with_kv_cache_mla(
            query=q,
            kv_cache=kv_cache,
            workspace_buffer=workspace_buffer,
            qk_nope_head_dim=config.qk_nope_head_dim,
            kv_lora_rank=config.kv_lora_rank,
            qk_rope_head_dim=config.qk_rope_head_dim,
            block_tables=block_tables,
            seq_lens=seq_lens,
            max_seq_len=max_seq_len,
            bmm1_scale=scale,
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


@register_backend("flashmla_decode")
def benchmark_flashmla_decode(
    config: MLAConfig,
    num_seqs: int,
    seq_len: int,
    query_len: int,
    num_warmup: int = 10,
    num_iters: int = 100,
) -> float:
    """Benchmark FlashMLA decode with FP8 KV cache."""
    from vllm.attention.ops.flashmla import (
        flash_mla_with_kvcache,
        get_mla_metadata,
        is_flashmla_dense_supported,
    )

    is_supported, reason = is_flashmla_dense_supported()
    if not is_supported:
        raise RuntimeError(f"FlashMLA dense not supported: {reason}")

    device = "cuda"

    # Create query tensor: (batch_size, seq_len_q, num_heads_q, head_dim)
    q = torch.randn(
        num_seqs,
        query_len,
        config.num_heads,
        config.head_size,
        device=device,
        dtype=config.dtype,
    )

    # Create block tables and sequence lengths
    block_tables, seq_lens, total_blocks = create_block_tables(
        num_seqs, seq_len, config.block_size, device
    )

    # Create KV cache: (num_blocks, page_block_size, num_heads_k=1, head_dim)
    kv_cache = torch.randn(
        block_tables.numel(),
        config.block_size,
        1,
        config.head_size,
        device=device,
        dtype=config.dtype,
    )

    # Get metadata for tile scheduling
    num_q_tokens_per_head_k = query_len * config.num_heads // 1  # MQA
    tile_scheduler_metadata, num_splits = get_mla_metadata(
        seq_lens,
        num_q_tokens_per_head_k,
        1,  # num_heads_k (MQA)
        is_fp8_kvcache=False,
    )

    # Compute scale
    scale = config.head_size**-0.5

    # Create descale tensors (identity for BF16)
    descale_q = torch.ones(1, device=device, dtype=torch.float32)
    descale_k = torch.ones(1, device=device, dtype=torch.float32)

    def run_kernel():
        return flash_mla_with_kvcache(
            q=q,
            k_cache=kv_cache,
            block_table=block_tables,
            cache_seqlens=seq_lens,
            head_dim_v=config.kv_lora_rank,
            tile_scheduler_metadata=tile_scheduler_metadata,
            num_splits=num_splits,
            softmax_scale=scale,
            causal=True,
            descale_q=descale_q,
            descale_k=descale_k,
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


@register_backend("flashmla_sparse_prefill")
def benchmark_flashmla_sparse_prefill(
    config: MLAConfig,
    num_seqs: int,
    seq_len: int,
    query_len: int,
    num_warmup: int = 10,
    num_iters: int = 100,
) -> float:
    """Benchmark FlashMLA sparse prefill."""
    from vllm.attention.ops.flashmla import (
        flash_mla_sparse_prefill,
        is_flashmla_sparse_supported,
    )

    is_supported, reason = is_flashmla_sparse_supported()
    if not is_supported:
        raise RuntimeError(f"FlashMLA sparse not supported: {reason}")

    device = "cuda"

    # Total tokens for prefill
    s_q = num_seqs * seq_len
    s_kv = s_q  # Same for prefill
    h_q = config.num_heads
    h_kv = 1  # MQA
    d_qk = config.head_size
    d_v = config.kv_lora_rank
    topk = 128  # Standard sparse attention top-k

    # Create tensors
    q = torch.randn(s_q, h_q, d_qk, device=device, dtype=config.dtype)
    kv = torch.randn(s_kv, h_kv, d_qk, device=device, dtype=config.dtype)

    # Create sparse indices: attend to first topk tokens for each query
    indices = torch.zeros(s_q, h_kv, topk, dtype=torch.int32, device=device)
    for i in range(s_q):
        # Valid indices are 0 to min(i, topk-1), rest are -1 (invalid)
        valid_len = min(i + 1, topk)
        if valid_len > 0:
            indices[i, 0, :valid_len] = torch.arange(
                max(0, i + 1 - topk), i + 1, device=device, dtype=torch.int32
            )
        if valid_len < topk:
            indices[i, 0, valid_len:] = -1

    scale = d_qk**-0.5

    def run_kernel():
        return flash_mla_sparse_prefill(q, kv, indices, scale, d_v)

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


@register_backend("trtllm_ragged_prefill")
def benchmark_trtllm_ragged_prefill(
    config: MLAConfig,
    num_seqs: int,
    seq_len: int,
    query_len: int,
    num_warmup: int = 10,
    num_iters: int = 100,
) -> float:
    """Benchmark TRT-LLM ragged attention prefill for DeepSeek MLA."""
    from flashinfer.prefill import trtllm_ragged_attention_deepseek

    device = "cuda"

    # Total tokens for prefill (causal self-attention)
    total_tokens = num_seqs * seq_len

    # DeepSeek-R1 MLA dimensions:
    # - q/k head_dim = qk_nope + qk_rope = 192
    # - v head_dim = v_head_dim = 128
    qk_head_dim = config.qk_head_dim  # 192
    v_head_dim = config.v_head_dim  # 128

    # Create tensors with MLA dimensions
    # q, k: [num_tokens, num_heads, qk_head_dim]
    # v: [num_tokens, num_heads, v_head_dim]
    q = torch.randn(
        total_tokens, config.num_heads, qk_head_dim, device=device, dtype=config.dtype
    )
    k = torch.randn(
        total_tokens, config.num_heads, qk_head_dim, device=device, dtype=config.dtype
    )
    v = torch.randn(
        total_tokens, config.num_heads, v_head_dim, device=device, dtype=config.dtype
    )

    # Create sequence lengths and cumulative sequence lengths
    seq_lens = torch.full((num_seqs,), seq_len, dtype=torch.int32, device=device)
    cum_seq_lens = torch.zeros(num_seqs + 1, dtype=torch.int32, device=device)
    cum_seq_lens[1:] = torch.cumsum(seq_lens, dim=0)

    # Create workspace buffer
    workspace_buffer = torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device=device)

    # Compute scale based on qk_head_dim
    scale = qk_head_dim**-0.5

    def run_kernel():
        return trtllm_ragged_attention_deepseek(
            query=q,
            key=k,
            value=v,
            workspace_buffer=workspace_buffer,
            seq_lens=seq_lens,
            max_q_len=seq_len,
            max_kv_len=seq_len,
            bmm1_scale=scale,
            bmm2_scale=1.0,
            o_sf_scale=1.0,
            batch_size=num_seqs,
            window_left=-1,
            cum_seq_lens_q=cum_seq_lens,
            cum_seq_lens_kv=cum_seq_lens,
            enable_pdl=False,
            is_causal=True,
            return_lse=False,
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


def run_benchmark(
    backend: str,
    config: MLAConfig,
    num_seqs: int,
    seq_len: int,
    query_len: int,
    num_warmup: int = 10,
    num_iters: int = 100,
) -> BenchmarkResult:
    """Run a single benchmark configuration."""
    if backend not in BACKEND_REGISTRY:
        raise ValueError(
            f"Unknown backend: {backend}. Available: {list(BACKEND_REGISTRY.keys())}"
        )

    total_tokens = num_seqs * query_len
    benchmark_fn = BACKEND_REGISTRY[backend]
    latency_us = benchmark_fn(
        config, num_seqs, seq_len, query_len, num_warmup, num_iters
    )

    throughput = total_tokens / (latency_us / 1e6)
    phase = "decode" if query_len == 1 else "prefill"

    return BenchmarkResult(
        backend=backend,
        num_seqs=num_seqs,
        seq_len=seq_len,
        query_len=query_len,
        total_tokens=total_tokens,
        num_heads=config.num_heads,
        kv_lora_rank=config.kv_lora_rank,
        block_size=config.block_size,
        kv_cache_dtype=config.kv_cache_dtype,
        latency_us=latency_us,
        throughput_tokens_s=throughput,
        phase=phase,
    )


def print_results_table(results: list[BenchmarkResult], title: str):
    """Print results in a formatted table."""
    print("=" * 110)
    print(title)
    print("=" * 110)
    print(
        f"{'Backend':<25} | {'Phase':<8} | {'Num Seqs':>10} | {'Seq Len':>8} | "
        f"{'Q Len':>6} | {'Latency (us)':>14} | {'Throughput (tok/s)':>18}"
    )
    print("-" * 110)

    for r in results:
        print(
            f"{r.backend:<25} | {r.phase:<8} | {r.num_seqs:>10} | {r.seq_len:>8} | "
            f"{r.query_len:>6} | {r.latency_us:>14.2f} | "
            f"{r.throughput_tokens_s:>18,.0f}"
        )
    print("=" * 110)


def save_results_csv(results: list[BenchmarkResult], filepath: str):
    """Save results to CSV file."""
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "backend",
                "phase",
                "num_seqs",
                "seq_len",
                "query_len",
                "total_tokens",
                "num_heads",
                "kv_lora_rank",
                "block_size",
                "kv_cache_dtype",
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
                    r.seq_len,
                    r.query_len,
                    r.total_tokens,
                    r.num_heads,
                    r.kv_lora_rank,
                    r.block_size,
                    r.kv_cache_dtype,
                    r.latency_us,
                    r.throughput_tokens_s,
                ]
            )
    print(f"Results saved to: {filepath}")


def save_results_json(results: list[BenchmarkResult], filepath: str):
    """Save results to JSON file."""
    data = {
        "benchmark_name": "flashinfer_mla",
        "model": "DeepSeek-R1",
        "device": torch.cuda.get_device_name(0),
        "results": [
            {
                "backend": r.backend,
                "phase": r.phase,
                "num_seqs": r.num_seqs,
                "seq_len": r.seq_len,
                "query_len": r.query_len,
                "total_tokens": r.total_tokens,
                "num_heads": r.num_heads,
                "kv_lora_rank": r.kv_lora_rank,
                "block_size": r.block_size,
                "kv_cache_dtype": r.kv_cache_dtype,
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

    # Color palette for different seq_lens
    colors = plt.cm.viridis([0.2, 0.5, 0.8, 0.95])

    for (backend, phase), phase_results in grouped.items():
        if not phase_results:
            continue

        # Create figure with two subplots: latency and throughput
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        if phase == "decode":
            # For decode: x-axis is num_seqs, different lines for seq_len
            seq_lens = sorted(set(r.seq_len for r in phase_results))

            for idx, seq_len in enumerate(seq_lens):
                data = [r for r in phase_results if r.seq_len == seq_len]
                data.sort(key=lambda x: x.num_seqs)

                num_seqs = [r.num_seqs for r in data]
                latencies = [r.latency_us for r in data]
                throughputs = [r.throughput_tokens_s / 1000 for r in data]  # K tok/s

                color = colors[idx % len(colors)]
                label = f"seq_len={seq_len}"

                axes[0].plot(
                    num_seqs, latencies, "o-", color=color, label=label, linewidth=2
                )
                axes[1].plot(
                    num_seqs, throughputs, "o-", color=color, label=label, linewidth=2
                )

            axes[0].set_xlabel("Number of Sequences (Batch Size)", fontsize=12)
            axes[1].set_xlabel("Number of Sequences (Batch Size)", fontsize=12)

        else:  # prefill
            # For prefill: x-axis is total_tokens, different lines for seq_len
            seq_lens = sorted(set(r.seq_len for r in phase_results))

            for idx, seq_len in enumerate(seq_lens):
                data = [r for r in phase_results if r.seq_len == seq_len]
                data.sort(key=lambda x: x.total_tokens)

                total_tokens = [r.total_tokens for r in data]
                latencies = [r.latency_us for r in data]
                throughputs = [r.throughput_tokens_s / 1e6 for r in data]  # M tok/s

                color = colors[idx % len(colors)]
                label = f"seq_len={seq_len}"

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

            axes[0].set_xlabel("Total Tokens", fontsize=12)
            axes[1].set_xlabel("Total Tokens", fontsize=12)

        # Configure latency subplot
        axes[0].set_ylabel("Latency (μs)", fontsize=12)
        axes[0].set_title(f"{backend} - Latency vs Batch Size", fontsize=14)
        axes[0].set_xscale("log", base=2)
        axes[0].set_ylim(bottom=0)
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(loc="best", fontsize=10)

        # Configure throughput subplot
        throughput_unit = "K tok/s" if phase == "decode" else "M tok/s"
        axes[1].set_ylabel(f"Throughput ({throughput_unit})", fontsize=12)
        axes[1].set_title(f"{backend} - Throughput vs Batch Size", fontsize=14)
        axes[1].set_xscale("log", base=2)
        axes[1].set_ylim(bottom=0)
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(loc="best", fontsize=10)

        # Add overall title
        fig.suptitle(
            f"MLA {phase.capitalize()} Benchmark - {backend}\n"
            f"Device: {device_name} | Model: DeepSeek-R1",
            fontsize=14,
            fontweight="bold",
        )
        plt.tight_layout()

        # Save figure
        filename = f"mla_{backend}_{phase}.png"
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

            # Use the first seq_len for comparison
            seq_lens = sorted(set(r.seq_len for r in phase_results))
            if not seq_lens:
                continue
            target_seq_len = seq_lens[0]

            data = [r for r in phase_results if r.seq_len == target_seq_len]
            data.sort(key=lambda x: x.num_seqs if phase == "decode" else x.total_tokens)

            if phase == "decode":
                x_values = [r.num_seqs for r in data]
                throughputs = [r.throughput_tokens_s / 1000 for r in data]
            else:
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

        fig.suptitle(
            f"MLA {phase.capitalize()} Backend Comparison\n"
            f"Device: {device_name} | seq_len={target_seq_len}",
            fontsize=14,
            fontweight="bold",
        )
        plt.tight_layout()

        filename = f"mla_comparison_{phase}.png"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Figure saved to: {filepath}")


def main():
    parser = FlexibleArgumentParser(
        description="Benchmark FlashInfer MLA backends (FP8 focus)"
    )
    parser.add_argument(
        "--backends",
        nargs="+",
        type=str,
        default=["flashmla_decode"],
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
        "--decode-seq-lens",
        nargs="+",
        type=int,
        default=[1024, 4096, 16384],
        help="KV cache sequence lengths for decode benchmarks",
    )
    parser.add_argument(
        "--decode-query-lens",
        nargs="+",
        type=int,
        default=[1],
        help="Query lengths for decode benchmarks (1 for decode, >1 for spec decode)",
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
        "--num-heads",
        type=int,
        default=128,
        help="Number of query heads (DeepSeek-R1: 128)",
    )
    parser.add_argument(
        "--kv-lora-rank",
        type=int,
        default=512,
        help="KV LoRA rank (DeepSeek-R1: 512)",
    )
    parser.add_argument(
        "--qk-nope-head-dim",
        type=int,
        default=128,
        help="QK nope head dimension (DeepSeek-R1: 128)",
    )
    parser.add_argument(
        "--qk-rope-head-dim",
        type=int,
        default=64,
        help="QK rope head dimension (DeepSeek-R1: 64)",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=64,
        help="KV cache block size",
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
    capability = current_platform.get_device_capability()
    if capability[0] < 9:
        print(
            "WARNING: FlashMLA requires SM90+ (Hopper/Blackwell). "
            "Some backends may not work."
        )
    if capability[0] < 10:
        print(
            "WARNING: FlashInfer MLA decode requires SM100 (Blackwell). "
            "flashinfer_mla_decode backend may not work."
        )

    # Create config
    config = MLAConfig(
        num_heads=args.num_heads,
        kv_lora_rank=args.kv_lora_rank,
        qk_nope_head_dim=args.qk_nope_head_dim,
        qk_rope_head_dim=args.qk_rope_head_dim,
        block_size=args.block_size,
    )

    print(f"MLA Configuration: {config}")
    print(f"Backends: {args.backends}")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print()

    results: list[BenchmarkResult] = []

    # Determine which backends support which modes
    decode_backends = [b for b in args.backends if "decode" in b]
    prefill_backends = [b for b in args.backends if "prefill" in b]

    # Run decode benchmarks
    if not args.prefill_only and decode_backends:
        print("Running DECODE benchmarks...")
        for backend in decode_backends:
            for seq_len in args.decode_seq_lens:
                for query_len in args.decode_query_lens:
                    for num_seqs in args.decode_num_seqs:
                        try:
                            result = run_benchmark(
                                backend=backend,
                                config=config,
                                num_seqs=num_seqs,
                                seq_len=seq_len,
                                query_len=query_len,
                                num_warmup=args.num_warmup,
                                num_iters=args.num_iters,
                            )
                            results.append(result)
                            print(
                                f"  {backend} | num_seqs={num_seqs:>5} | "
                                f"seq_len={seq_len:>6} | q_len={query_len:>2} | "
                                f"latency={result.latency_us:>10.2f} us | "
                                f"throughput={result.throughput_tokens_s:>12,.0f} tok/s"
                            )
                        except Exception as e:
                            print(
                                f"  {backend} | num_seqs={num_seqs:>5} | "
                                f"seq_len={seq_len:>6} | "
                                f"q_len={query_len:>2} | ERROR: {e}"
                            )

    # Run prefill benchmarks
    if not args.decode_only and prefill_backends:
        print("\nRunning PREFILL benchmarks...")
        for backend in prefill_backends:
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
                            seq_len=seq_len,
                            query_len=seq_len,  # For prefill, query_len == seq_len
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
            f"FlashInfer MLA Benchmark Results - {torch.cuda.get_device_name(0)}",
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
    main()
