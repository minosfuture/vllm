#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Benchmark script for MoE All2All dispatch and combine latency.

Measures dispatch and combine communication overhead separately for all
available all2all backends used in MoE expert parallelism.  No expert
GEMM computation is included.

Usage:
    python benchmarks/kernels/benchmark_all2all.py \
        --model nvidia/DeepSeek-R1-0528-FP4-v2 --num-gpus 2

Available backends: naive, pplx, deepep_high_throughput, deepep_low_latency,
                    mori, flashinfer_all2allv
"""

import dataclasses
import json
import os
import traceback
from collections.abc import Callable

import torch
import torch.distributed as dist
from torch.multiprocessing import spawn  # pyright: ignore[reportPrivateImportUsage]

from vllm.logger import init_logger
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.utils.import_utils import has_deep_ep, has_mori, has_pplx
from vllm.utils.network_utils import get_open_port

logger = init_logger(__name__)

DEFAULT_NUM_TOKENS = [1, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]

ALL_BACKENDS = [
    "naive",
    "allgather_reducescatter",
    "pplx",
    "deepep_high_throughput",
    "deepep_low_latency",
    "mori",
    "flashinfer_all2allv",
]


# ---------------------------------------------------------------------------
# Model config extraction (reused from benchmark_moe.py)
# ---------------------------------------------------------------------------


def get_model_params(config):
    """Extract num_experts, topk, intermediate_size, hidden_size from HF config."""
    if config.architectures[0] == "DbrxForCausalLM":
        E = config.ffn_config.moe_num_experts
        topk = config.ffn_config.moe_top_k
        intermediate_size = config.ffn_config.ffn_hidden_size
        hidden_size = config.hidden_size
    elif config.architectures[0] == "JambaForCausalLM":
        E = config.num_experts
        topk = config.num_experts_per_tok
        intermediate_size = config.intermediate_size
        hidden_size = config.hidden_size
    elif config.architectures[0] in (
        "DeepseekV2ForCausalLM",
        "DeepseekV3ForCausalLM",
        "DeepseekV32ForCausalLM",
        "Glm4MoeForCausalLM",
        "Glm4MoeLiteForCausalLM",
        "NemotronHForCausalLM",
        "MistralLarge3ForCausalLM",
    ):
        E = config.n_routed_experts
        topk = config.num_experts_per_tok
        intermediate_size = config.moe_intermediate_size
        hidden_size = config.hidden_size
    elif config.architectures[0] in (
        "Qwen2MoeForCausalLM",
        "Qwen3MoeForCausalLM",
        "Qwen3NextForCausalLM",
    ):
        E = config.num_experts
        topk = config.num_experts_per_tok
        intermediate_size = config.moe_intermediate_size
        hidden_size = config.hidden_size
    elif config.architectures[0] == "Qwen3VLMoeForConditionalGeneration":
        text_config = config.get_text_config()
        E = text_config.num_experts
        topk = text_config.num_experts_per_tok
        intermediate_size = text_config.moe_intermediate_size
        hidden_size = text_config.hidden_size
    elif config.architectures[0] == "HunYuanMoEV1ForCausalLM":
        E = config.num_experts
        topk = config.moe_topk[0]
        intermediate_size = config.moe_intermediate_size[0]
        hidden_size = config.hidden_size
    elif config.architectures[0] == "Qwen3OmniMoeForConditionalGeneration":
        E = config.thinker_config.text_config.num_experts
        topk = config.thinker_config.text_config.num_experts_per_tok
        intermediate_size = config.thinker_config.text_config.moe_intermediate_size
        hidden_size = config.thinker_config.text_config.hidden_size
    elif config.architectures[0] == "PixtralForConditionalGeneration":
        return get_model_params(config.get_text_config())
    else:
        config = config.get_text_config()
        E = config.num_local_experts
        topk = config.num_experts_per_tok
        intermediate_size = config.intermediate_size
        hidden_size = config.hidden_size
    return E, topk, intermediate_size, hidden_size


# ---------------------------------------------------------------------------
# Process group info
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class ProcessGroupInfo:
    world_size: int
    rank: int
    local_rank: int
    device: torch.device


# ---------------------------------------------------------------------------
# Backend availability checking
# ---------------------------------------------------------------------------


def _has_flashinfer_all2all() -> bool:
    try:
        from flashinfer.comm.trtllm_alltoall import MnnvlMoe  # noqa: F401

        return True
    except ImportError:
        return False


def get_available_backends(requested: list[str] | None) -> list[str]:
    """Return list of backends that are both requested and importable."""
    checks: dict[str, Callable[[], bool]] = {
        "naive": lambda: True,  # always available
        "allgather_reducescatter": lambda: True,  # always available
        "pplx": has_pplx,
        "deepep_high_throughput": has_deep_ep,
        "deepep_low_latency": has_deep_ep,
        "mori": has_mori,
        "flashinfer_all2allv": _has_flashinfer_all2all,
    }

    candidates = requested if requested else ALL_BACKENDS
    available: list[str] = []
    for backend in candidates:
        if backend not in checks:
            print(f"[WARN] Unknown backend '{backend}', skipping.")
            continue
        try:
            if checks[backend]():
                # mori only works on ROCm
                if backend == "mori":
                    from vllm.platforms import current_platform

                    if not current_platform.is_rocm():
                        print(f"[INFO] Skipping '{backend}' (ROCm only).")
                        continue
                available.append(backend)
            else:
                print(f"[INFO] Backend '{backend}' not available (import failed).")
        except Exception as e:
            print(f"[INFO] Backend '{backend}' not available: {e}")
    return available


# ---------------------------------------------------------------------------
# Forward context mocking for naive / allgather_reducescatter backends
# ---------------------------------------------------------------------------


def _setup_forward_context(num_tokens: int, world_size: int):
    """
    Create a minimal forward context with DPMetadata so that
    NaiveAll2AllManager / AgRsAll2AllManager can function.
    """
    import vllm.forward_context as fc_mod
    from vllm.forward_context import DPMetadata, ForwardContext

    num_tokens_across_dp = torch.tensor([num_tokens] * world_size, dtype=torch.int64)
    dp_metadata = DPMetadata(
        max_tokens_across_dp_cpu=torch.tensor(num_tokens, dtype=torch.int64),
        num_tokens_across_dp_cpu=num_tokens_across_dp,
        local_sizes=[num_tokens] * world_size,
    )

    ctx = ForwardContext(
        no_compile_layers={},
        all_moe_layers=None,
        attn_metadata={},
        slot_mapping={},
        virtual_engine=0,
        dp_metadata=dp_metadata,
    )
    fc_mod._forward_context = ctx


def _ensure_naive_all2all_manager(pgi: ProcessGroupInfo, variant: str):
    """
    Ensure the EP group's device communicator has the right all2all manager
    for the naive / allgather_reducescatter backends.
    """
    from vllm.distributed import get_ep_group

    ep = get_ep_group()
    dc = ep.device_communicator

    if variant == "naive":
        from vllm.distributed.device_communicators.all2all import (
            NaiveAll2AllManager,
        )

        if not isinstance(dc.all2all_manager, NaiveAll2AllManager):
            dc.all2all_manager = NaiveAll2AllManager(dc.cpu_group)
    else:
        from vllm.distributed.device_communicators.all2all import (
            AgRsAll2AllManager,
        )

        if not isinstance(dc.all2all_manager, AgRsAll2AllManager):
            dc.all2all_manager = AgRsAll2AllManager(dc.cpu_group)


# ---------------------------------------------------------------------------
# Backend factories
# ---------------------------------------------------------------------------


def make_naive_a2a(pgi: ProcessGroupInfo, num_experts: int, **kwargs):
    """Create MoEPrepareAndFinalizeNaiveEP (uses naive broadcast all2all)."""
    from vllm.model_executor.layers.fused_moe.prepare_finalize import (
        MoEPrepareAndFinalizeNaiveEP,
    )

    _ensure_naive_all2all_manager(pgi, "naive")

    # is_sequence_parallel=True routes dispatch/combine through the EP group
    # (which spans all ranks) rather than the DP group (size 1 in benchmark).
    return MoEPrepareAndFinalizeNaiveEP(
        is_sequence_parallel=True,
        num_dispatchers=pgi.world_size,
    )


def make_allgather_reducescatter_a2a(pgi: ProcessGroupInfo, num_experts: int, **kwargs):
    """Create MoEPrepareAndFinalizeNaiveEP (uses allgather/reducescatter)."""
    from vllm.model_executor.layers.fused_moe.prepare_finalize import (
        MoEPrepareAndFinalizeNaiveEP,
    )

    _ensure_naive_all2all_manager(pgi, "allgather_reducescatter")

    return MoEPrepareAndFinalizeNaiveEP(
        is_sequence_parallel=True,
        num_dispatchers=pgi.world_size,
    )


def make_pplx_a2a(
    pgi: ProcessGroupInfo,
    num_experts: int,
    topk: int,
    hidden_size: int,
    max_num_tokens: int,
    dtype: torch.dtype,
    **kwargs,
):
    """Create PplxPrepareAndFinalize via pplx_kernels.AllToAll.intranode()."""
    from pplx_kernels import AllToAll

    from vllm.model_executor.layers.fused_moe.pplx_prepare_finalize import (
        PplxPrepareAndFinalize,
        pplx_hidden_dim_scale_bytes,
    )

    num_local_experts = num_experts // pgi.world_size

    hidden_dim_bytes, scale_bytes = pplx_hidden_dim_scale_bytes(
        max_num_tokens,
        hidden_size,
        dtype,
        None,  # no quant
        per_act_token_quant=False,
        block_shape=None,
    )

    group_name = dist.group.WORLD.group_name

    ata = AllToAll.intranode(
        max_num_tokens=max_num_tokens,
        num_experts=num_experts,
        experts_per_token=topk,
        rank=pgi.rank,
        world_size=pgi.world_size,
        dp_size=1,
        hidden_dim=hidden_size,
        hidden_dim_bytes=hidden_dim_bytes,
        hidden_dim_scale_bytes=scale_bytes,
        group_name=group_name,
    )

    pf = PplxPrepareAndFinalize(
        ata,
        max_num_tokens=max_num_tokens,
        num_local_experts=num_local_experts,
        num_dispatchers=pgi.world_size,
    )
    return pf


def make_deepep_ht_a2a(
    pgi: ProcessGroupInfo,
    num_experts: int,
    **kwargs,
):
    """Create DeepEPHTPrepareAndFinalize."""
    import deep_ep

    from vllm.model_executor.layers.fused_moe.deepep_ht_prepare_finalize import (
        DeepEPHTPrepareAndFinalize,
    )

    pg = dist.group.WORLD
    num_local_experts = num_experts // pgi.world_size

    buffer = deep_ep.Buffer(
        group=pg,
        num_nvl_bytes=1024 * 1024 * 1024,  # 1GB
        num_rdma_bytes=0,
        low_latency_mode=False,
        num_qps_per_rank=1,
    )

    return DeepEPHTPrepareAndFinalize(
        buffer=buffer,
        num_dispatchers=pgi.world_size,
        dp_size=1,
        rank_expert_offset=pgi.rank * num_local_experts,
    )


def make_deepep_ll_a2a(
    pgi: ProcessGroupInfo,
    num_experts: int,
    hidden_size: int,
    max_num_tokens: int,
    **kwargs,
):
    """Create DeepEPLLPrepareAndFinalize."""
    import deep_ep

    from vllm.model_executor.layers.fused_moe.deepep_ll_prepare_finalize import (
        DeepEPLLPrepareAndFinalize,
    )

    pg = dist.group.WORLD

    # Round up hidden_size to supported value
    hidden_size = DeepEPLLPrepareAndFinalize.maybe_roundup_layer_hidden_size(
        hidden_size
    )

    num_rdma_bytes = deep_ep.Buffer.get_low_latency_rdma_size_hint(
        max_num_tokens,
        hidden_size,
        pgi.world_size,
        num_experts,
    )

    buffer = deep_ep.Buffer(
        group=pg,
        num_rdma_bytes=num_rdma_bytes,
        low_latency_mode=True,
        num_qps_per_rank=num_experts // pgi.world_size,
    )

    return DeepEPLLPrepareAndFinalize(
        buffer=buffer,
        max_tokens_per_rank=max_num_tokens,
        num_dispatchers=pgi.world_size,
        use_fp8_dispatch=False,
    )


def make_mori_a2a(
    pgi: ProcessGroupInfo,
    num_experts: int,
    topk: int,
    hidden_size: int,
    max_num_tokens: int,
    dtype: torch.dtype,
    **kwargs,
):
    """Create MoriPrepareAndFinalize."""
    import mori

    from vllm.model_executor.layers.fused_moe.mori_prepare_finalize import (
        MoriPrepareAndFinalize,
    )

    num_local_experts = num_experts // pgi.world_size

    mori_op = mori.ops.EpDispatchCombineOp(
        rank=pgi.rank,
        num_ep_ranks=pgi.world_size,
        quant_dtype=None,
        token_hidden_size=hidden_size,
        scale_dim=1,
        scale_type_size=torch.float32.itemsize,
        max_num_tokens_per_dp_rank=max_num_tokens,
        input_dtype=dtype,
        num_local_experts=num_local_experts,
        num_experts_per_token=topk,
    )

    return MoriPrepareAndFinalize(
        mori_op,
        max_tokens_per_rank=max_num_tokens,
        num_dispatchers=pgi.world_size,
        use_fp8_dispatch=False,
    )


def make_flashinfer_a2a(
    pgi: ProcessGroupInfo,
    num_experts: int,
    num_tokens: int,
    **kwargs,
):
    """Create FlashInferA2APrepareAndFinalize with mocked get_local_sizes."""
    import vllm.model_executor.layers.fused_moe.flashinfer_a2a_prepare_finalize as fi_mod  # noqa: E501
    from vllm.distributed import get_ep_group
    from vllm.distributed.device_communicators.all2all import (
        FlashInferAllToAllManager,
    )
    from vllm.model_executor.layers.fused_moe.flashinfer_a2a_prepare_finalize import (
        FlashInferA2APrepareAndFinalize,
    )

    world_size = pgi.world_size
    ep = get_ep_group()
    dc = ep.device_communicator

    if dc.all2all_manager is None or not isinstance(
        dc.all2all_manager, FlashInferAllToAllManager
    ):
        mgr = FlashInferAllToAllManager(dc.cpu_group)

        # Override initialize to use the EP group communicator for MNNVL
        # workspace allocation instead of the DP group (which is size 1
        # in our benchmark setup).
        _orig_initialize = mgr.initialize

        def _patched_initialize(world_size, rank, gpus_per_node):
            from flashinfer.comm.trtllm_alltoall import Mapping, MnnvlConfig, MnnvlMoe

            from vllm.distributed.device_communicators.mnnvl_compat import (
                CustomCommunicator,
            )

            if mgr.initialized:
                return
            mgr.cleanup()
            mgr.mapping = Mapping(
                world_size,
                rank,
                gpus_per_node,
                tp_size=world_size,
            )
            # Use the EP group's cpu_group (all ranks) instead of DP group
            dp_config = MnnvlConfig(
                comm_backend=CustomCommunicator(ep.cpu_group),
                fabric_page_size=1 << 29,
                allocation_granularity=0,
            )
            mgr.workspace_tensor = MnnvlMoe.get_moe_workspaces(mgr.mapping, dp_config)
            mgr.prepare_workspace_tensor = MnnvlMoe.get_moe_prepare_workspace(
                mgr.mapping, dp_config
            )
            mgr.world_size = world_size
            mgr.rank = rank
            mgr.gpus_per_node = gpus_per_node
            mgr.initialized = True

        mgr.initialize = _patched_initialize
        dc.all2all_manager = mgr

    # Monkey-patch get_local_sizes to return synthetic sizes
    fi_mod.get_local_sizes = lambda: [num_tokens] * world_size

    pf = FlashInferA2APrepareAndFinalize(
        num_dispatchers=world_size,
    )

    # Ensure workspace is initialized
    pf.all2all_manager.ensure_alltoall_workspace_initialized()

    return pf


# ---------------------------------------------------------------------------
# Timing utility
# ---------------------------------------------------------------------------


def _make_expert_output(
    dispatched_a: torch.Tensor | None,
    topk_ids: torch.Tensor,
    topk: int,
    hidden_size: int,
    dtype: torch.dtype,
    device: torch.device,
    backend_name: str,
) -> torch.Tensor:
    """
    Create dummy expert output tensor matching the shape expected by finalize.

    Standard-format backends (naive, deepep_ht, flashinfer):
      Expert output is (M_dispatched * topk, hidden_size).
    Batched-format backends (pplx, deepep_ll, mori):
      Expert output has the same shape as dispatched_a
      (e.g. (E, max_tokens, hidden) or (E, max_tokens * dispatchers, hidden)).
    """
    if dispatched_a is None:
        # Fallback: shouldn't happen, but be safe
        return torch.randn(
            topk_ids.shape[0] * topk, hidden_size, dtype=dtype, device=device
        )

    batched_backends = {"pplx", "deepep_low_latency", "mori"}
    if backend_name in batched_backends:
        # Batched format: expert output matches dispatched_a shape
        return torch.randn_like(dispatched_a).to(dtype)
    else:
        # Standard format: (M_dispatched * topk, hidden_size)
        m_dispatched = dispatched_a.shape[0]
        return torch.randn(m_dispatched * topk, hidden_size, dtype=dtype, device=device)


def benchmark_one(
    prepare_finalize,
    a1: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    num_experts: int,
    hidden_size: int,
    topk: int,
    quant_config,
    num_warmup: int,
    num_trials: int,
    backend_name: str,
) -> dict[str, float]:
    """
    Benchmark dispatch (prepare) and combine (finalize) separately.
    Returns dict with 'dispatch_ms', 'combine_ms', 'total_ms'.
    """
    from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
        TopKWeightAndReduceContiguous,
        TopKWeightAndReduceDelegate,
    )

    weight_and_reduce = TopKWeightAndReduceContiguous()

    # Backends whose finalize asserts TopKWeightAndReduceDelegate
    # (weight application happens inside the combine kernel)
    delegate_backends = {"pplx", "deepep_low_latency"}
    if backend_name in delegate_backends:
        weight_and_reduce = TopKWeightAndReduceDelegate()

    device = a1.device
    output = torch.zeros(a1.shape[0], hidden_size, dtype=a1.dtype, device=device)

    # FlashInfer alltoall dispatch always tries to dispatch scales alongside
    # activations.  With unquantized config (quant_dtype=None) the scales are
    # None, causing a crash.  Use defer_input_quant=True so it takes the
    # passthrough path that only dispatches activations.
    defer_input_quant = backend_name == "flashinfer_all2allv"

    # Cast topk_ids to required dtype
    required_dtype = prepare_finalize.topk_indices_dtype()
    if required_dtype is not None:
        topk_ids_cast = topk_ids.to(required_dtype)
    else:
        topk_ids_cast = topk_ids

    # --- Warmup ---
    for _ in range(num_warmup):
        result = prepare_finalize.prepare(
            a1.clone(),
            topk_weights.clone(),
            topk_ids_cast.clone(),
            num_experts,
            None,  # expert_map
            False,  # apply_router_weight_on_input
            quant_config,
            defer_input_quant,
        )
        dispatched_a, dispatched_scale, expert_meta, disp_ids, disp_weights = result

        expert_out = _make_expert_output(
            dispatched_a,
            topk_ids_cast,
            topk,
            hidden_size,
            a1.dtype,
            device,
            backend_name,
        )
        fin_topk_weights = disp_weights if disp_weights is not None else topk_weights
        fin_topk_ids = disp_ids if disp_ids is not None else topk_ids_cast

        prepare_finalize.finalize(
            output,
            expert_out,
            fin_topk_weights,
            fin_topk_ids,
            False,  # apply_router_weight_on_input
            weight_and_reduce,
        )
    torch.cuda.synchronize()

    # --- Measure dispatch ---
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(num_trials):
        result = prepare_finalize.prepare(
            a1.clone(),
            topk_weights.clone(),
            topk_ids_cast.clone(),
            num_experts,
            None,
            False,
            quant_config,
            defer_input_quant,
        )
    end_event.record()
    torch.cuda.synchronize()
    dispatch_ms = start_event.elapsed_time(end_event) / num_trials

    # --- Measure combine ---
    # Warmup combine (each trial needs fresh dispatch for handle state)
    for _ in range(num_warmup):
        result = prepare_finalize.prepare(
            a1.clone(),
            topk_weights.clone(),
            topk_ids_cast.clone(),
            num_experts,
            None,
            False,
            quant_config,
            defer_input_quant,
        )
        da, ds, em, di, dw = result
        eo = _make_expert_output(
            da,
            topk_ids_cast,
            topk,
            hidden_size,
            a1.dtype,
            device,
            backend_name,
        )
        fw = dw if dw is not None else topk_weights
        fi = di if di is not None else topk_ids_cast
        prepare_finalize.finalize(output, eo, fw, fi, False, weight_and_reduce)
    torch.cuda.synchronize()

    # Timed combine: each trial needs a fresh dispatch for handle state
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(num_trials):
        # Dispatch to set up handle state
        result = prepare_finalize.prepare(
            a1.clone(),
            topk_weights.clone(),
            topk_ids_cast.clone(),
            num_experts,
            None,
            False,
            quant_config,
            defer_input_quant,
        )
        da, ds, em, di, dw = result
        eo = _make_expert_output(
            da,
            topk_ids_cast,
            topk,
            hidden_size,
            a1.dtype,
            device,
            backend_name,
        )
        fw = dw if dw is not None else topk_weights
        fi = di if di is not None else topk_ids_cast
        prepare_finalize.finalize(output, eo, fw, fi, False, weight_and_reduce)
    end_event.record()
    torch.cuda.synchronize()
    dispatch_plus_combine_ms = start_event.elapsed_time(end_event) / num_trials
    combine_ms = dispatch_plus_combine_ms - dispatch_ms

    return {
        "dispatch_ms": dispatch_ms,
        "combine_ms": combine_ms,
        "total_ms": dispatch_ms + combine_ms,
    }


# ---------------------------------------------------------------------------
# Worker function
# ---------------------------------------------------------------------------


def worker(
    local_rank: int,
    world_size: int,
    init_method: str,
    model_name: str,
    trust_remote_code: bool,
    backends: list[str],
    token_counts: list[int],
    dtype: torch.dtype,
    num_warmup: int,
    num_trials: int,
    max_num_tokens: int,
    output_json: str | None,
):
    """Worker process for benchmarking."""
    rank = local_rank
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    dist.init_process_group(
        backend="cpu:gloo,cuda:nccl",
        init_method=init_method,
        rank=rank,
        world_size=world_size,
        device_id=device,
    )
    barrier = torch.tensor([rank], device=device)
    dist.all_reduce(barrier)

    pgi = ProcessGroupInfo(
        world_size=world_size,
        rank=rank,
        local_rank=local_rank,
        device=device,
    )

    # Initialize vLLM distributed env (needed for naive EP, flashinfer, etc.)
    try:
        from vllm.distributed.parallel_state import (
            GroupCoordinator,
            init_distributed_environment,
            initialize_model_parallel,
        )

        # init_distributed_environment creates the _WORLD group that
        # initialize_model_parallel requires.  Since torch.distributed is
        # already initialized it will skip init_process_group and only
        # create the vLLM world group wrapper.
        init_distributed_environment(
            world_size=world_size,
            rank=rank,
            local_rank=local_rank,
            backend="nccl",
        )

        # Creates TP, PP, DP, and EP groups.
        # With no vllm config and tp_size=1, EP group ends up with size 1
        # per rank.  We fix this below.
        initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
        )

        # Override the EP group to span all ranks so that all2all backends
        # that rely on get_ep_group() have a proper multi-rank EP group
        # with a device communicator.
        import vllm.distributed.parallel_state as ps

        all_ranks = list(range(world_size))
        ps._EP = GroupCoordinator(
            group_ranks=[all_ranks],
            local_rank=local_rank,
            torch_distributed_backend="nccl",
            use_device_communicator=True,
            group_name="ep",
        )
    except Exception as e:
        if rank == 0:
            print(f"[WARN] Could not init vLLM distributed env: {e}")
            traceback.print_exc()

    # Load model config
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    num_experts, topk, intermediate_size, hidden_size = get_model_params(config)

    if rank == 0:
        print(f"\nModel: {model_name}")
        print(
            f"  num_experts={num_experts}, topk={topk}, "
            f"hidden_size={hidden_size}, intermediate_size={intermediate_size}"
        )
        print(f"  world_size={world_size}, dtype={dtype}")
        print(f"  max_num_tokens={max_num_tokens}")
        print(f"  backends={backends}")
        print()

    from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig

    quant_config = FusedMoEQuantConfig.make()

    # Factory map
    factory_map: dict[str, Callable] = {
        "naive": make_naive_a2a,
        "allgather_reducescatter": make_allgather_reducescatter_a2a,
        "pplx": make_pplx_a2a,
        "deepep_high_throughput": make_deepep_ht_a2a,
        "deepep_low_latency": make_deepep_ll_a2a,
        "mori": make_mori_a2a,
        "flashinfer_all2allv": make_flashinfer_a2a,
    }

    all_results: dict[str, dict[int, dict[str, float]]] = {}

    for backend_name in backends:
        if rank == 0:
            print(f"--- Backend: {backend_name} ---")

        backend_results: dict[int, dict[str, float]] = {}

        for num_tokens in token_counts:
            # Check max_num_tokens constraint for batched backends
            effective_max = max_num_tokens
            if (
                backend_name in ("pplx", "deepep_low_latency", "mori")
                and num_tokens > effective_max
            ):
                if rank == 0:
                    print(
                        f"  num_tokens={num_tokens}: SKIPPED "
                        f"(exceeds max_num_tokens={effective_max})"
                    )
                continue

            try:
                # Naive / AgRs backends need a mocked forward context
                if backend_name in ("naive", "allgather_reducescatter"):
                    _setup_forward_context(num_tokens, world_size)

                factory = factory_map[backend_name]
                factory_kwargs = dict(
                    pgi=pgi,
                    num_experts=num_experts,
                    topk=topk,
                    hidden_size=hidden_size,
                    max_num_tokens=max(num_tokens, effective_max),
                    dtype=dtype,
                    num_tokens=num_tokens,
                )

                pf = factory(**factory_kwargs)

                # Generate synthetic data
                a1 = torch.randn(num_tokens, hidden_size, dtype=dtype, device=device)
                topk_ids = torch.randint(
                    0,
                    num_experts,
                    (num_tokens, topk),
                    device=device,
                    dtype=torch.int32,
                )
                topk_weights = torch.softmax(
                    torch.randn(num_tokens, topk, device=device, dtype=torch.float32),
                    dim=-1,
                )  # keep float32 — required by DeepEP and PPLX kernels

                dist.barrier()

                result = benchmark_one(
                    prepare_finalize=pf,
                    a1=a1,
                    topk_weights=topk_weights,
                    topk_ids=topk_ids,
                    num_experts=num_experts,
                    hidden_size=hidden_size,
                    topk=topk,
                    quant_config=quant_config,
                    num_warmup=num_warmup,
                    num_trials=num_trials,
                    backend_name=backend_name,
                )

                backend_results[num_tokens] = result

                if rank == 0:
                    print(
                        f"  num_tokens={num_tokens:>5d}: "
                        f"dispatch={result['dispatch_ms']:>8.3f}ms  "
                        f"combine={result['combine_ms']:>8.3f}ms  "
                        f"total={result['total_ms']:>8.3f}ms"
                    )

            except Exception as e:
                if rank == 0:
                    print(f"  num_tokens={num_tokens}: ERROR - {e}")
                    traceback.print_exc()

            dist.barrier()

        all_results[backend_name] = backend_results

        if rank == 0:
            print()

    # Print summary table
    if rank == 0:
        print_summary_table(
            all_results,
            token_counts,
            backends,
            world_size,
            dtype,
            hidden_size,
            num_experts,
            topk,
        )

        if output_json:
            save_json(
                all_results,
                output_json,
                world_size,
                dtype,
                hidden_size,
                num_experts,
                topk,
                model_name,
                num_warmup,
                num_trials,
            )

    dist.barrier()
    dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------


def print_summary_table(
    all_results: dict[str, dict[int, dict[str, float]]],
    token_counts: list[int],
    backends: list[str],
    world_size: int,
    dtype: torch.dtype,
    hidden_size: int,
    num_experts: int,
    topk: int,
):
    """Print formatted summary table."""
    # Collect all token counts that have at least one result
    active_tokens = sorted({tc for r in all_results.values() for tc in r})
    if not active_tokens:
        print("No results to display.")
        return

    active_backends = [b for b in backends if b in all_results and all_results[b]]

    print(f"\n{'=' * 130}")
    print("MoE All2All Benchmark Results")
    print(
        f"  World Size: {world_size}, dtype: {dtype}, "
        f"hidden_size: {hidden_size}, num_experts: {num_experts}, topk: {topk}"
    )
    print(f"{'=' * 130}")

    # Print dispatch table
    _print_phase_table(
        "DISPATCH (ms)", active_backends, active_tokens, all_results, "dispatch_ms"
    )
    print()
    _print_phase_table(
        "COMBINE (ms)", active_backends, active_tokens, all_results, "combine_ms"
    )
    print()
    _print_phase_table(
        "TOTAL (ms)", active_backends, active_tokens, all_results, "total_ms"
    )
    print(f"{'=' * 130}")


def _print_phase_table(
    title: str,
    backends: list[str],
    token_counts: list[int],
    all_results: dict[str, dict[int, dict[str, float]]],
    key: str,
):
    """Print one phase (dispatch/combine/total) table."""
    col_w = max(22, max(len(b) for b in backends) + 2)
    header = f"{'num_tokens':>12}"
    for b in backends:
        header += f"  {b:>{col_w}}"
    print(f"\n  {title}")
    print(f"  {header}")
    print(f"  {'-' * len(header)}")

    for tc in token_counts:
        row = f"  {tc:>12d}"
        for b in backends:
            if tc in all_results.get(b, {}):
                val = all_results[b][tc][key]
                row += f"  {val:>{col_w}.3f}"
            else:
                row += f"  {'N/A':>{col_w}}"
        print(row)


def save_json(
    all_results: dict[str, dict[int, dict[str, float]]],
    output_path: str,
    world_size: int,
    dtype: torch.dtype,
    hidden_size: int,
    num_experts: int,
    topk: int,
    model_name: str,
    num_warmup: int,
    num_trials: int,
):
    """Save results to JSON file."""
    # Convert int keys to strings for JSON
    json_results = {}
    for backend, token_results in all_results.items():
        json_results[backend] = {str(k): v for k, v in token_results.items()}

    output_data = {
        "model": model_name,
        "world_size": world_size,
        "dtype": str(dtype),
        "hidden_size": hidden_size,
        "num_experts": num_experts,
        "topk": topk,
        "num_warmup": num_warmup,
        "num_trials": num_trials,
        "results": json_results,
    }

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"Results saved to {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = FlexibleArgumentParser(
        description="Benchmark MoE All2All dispatch and combine latency"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="nvidia/DeepSeek-R1-0528-FP4-v2",
        help="HuggingFace model name to extract MoE config from",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code when loading HF config",
    )
    parser.add_argument(
        "--backend",
        "--backends",
        type=str,
        nargs="+",
        default=None,
        choices=ALL_BACKENDS,
        dest="backends",
        help="Backends to benchmark (default: auto-detect all installed)",
    )
    parser.add_argument(
        "--num-tokens",
        type=int,
        nargs="+",
        default=DEFAULT_NUM_TOKENS,
        help="Token counts to benchmark",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16"],
        help="Data type for activations",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=None,
        help="Number of GPUs to use (default: auto-detect)",
    )
    parser.add_argument(
        "--num-warmup",
        type=int,
        default=5,
        help="Number of warmup iterations",
    )
    parser.add_argument(
        "--num-trials",
        type=int,
        default=20,
        help="Number of benchmark trials",
    )
    parser.add_argument(
        "--max-num-tokens",
        type=int,
        default=256,
        help="Max tokens per rank for batched backends (pplx, deepep_ll, mori)",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Path to save JSON results",
    )

    args = parser.parse_args()

    dtype = getattr(torch, args.dtype)
    num_gpus = args.num_gpus or torch.cuda.device_count()

    if num_gpus < 2:
        print("ERROR: At least 2 GPUs are required for all2all benchmarking.")
        return

    # Check available backends
    available = get_available_backends(args.backends)
    if not available:
        print("ERROR: No backends available for benchmarking.")
        return

    print(f"Available backends: {available}")
    print(f"Using {num_gpus} GPUs")

    port = get_open_port()
    init_method = f"tcp://{os.getenv('LOCALHOST', 'localhost')}:{port}"

    spawn(
        worker,
        args=(
            num_gpus,
            init_method,
            args.model,
            args.trust_remote_code,
            available,
            args.num_tokens,
            dtype,
            args.num_warmup,
            args.num_trials,
            args.max_num_tokens,
            args.output_json,
        ),
        nprocs=num_gpus,
        join=True,
    )


if __name__ == "__main__":
    main()
