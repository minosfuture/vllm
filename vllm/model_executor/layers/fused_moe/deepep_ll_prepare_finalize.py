# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass

import deep_ep
import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm import envs
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceDelegate,
)
from vllm.model_executor.layers.fused_moe.utils import (
    moe_kernel_quantize_input,
    normalize_batched_scales_shape,
)
from vllm.v1.worker.ubatching import (
    dbo_current_ubatch_id,
    dbo_enabled,
    dbo_maybe_run_recv_hook,
)

alt_stream = torch.cuda.Stream()

logger = init_logger(__name__)

# DeepEP kernels quantize dispatch inputs in 128 element chunks.
DEEPEP_QUANT_BLOCK_SIZE = 128
DEEPEP_QUANT_BLOCK_SHAPE = [DEEPEP_QUANT_BLOCK_SIZE, DEEPEP_QUANT_BLOCK_SIZE]

logger = init_logger(__name__)


@dataclass
class CombineOverlapArgs:
    # we launch deepep ll combine on this stream, which
    # is different from the default compute stream
    stream: torch.cuda.Stream
    # We record this wait even in the compute stream between
    # silu_mul_fp4_quantize and w2 gemm.
    # And we wait for this even before deepep ll combine on the
    # combine stream to ensure signal tensors have been allocated
    wait_event: torch.cuda.Event
    # Number of CU used for combine kernel, currently hardcoded to be 32
    num_sms: int
    # The signal tensor is shared by the w2 gemm and deepep ll combine.
    # w2 gemm atomic_add to the tensor to signal deepep combine can start
    # send data
    signal: torch.Tensor | None = None
    # Set to the number of CU used by W2 gemm, which is a persistent kernel
    # So when all CU has completed the computation for an expert,
    # combine kernel can start to send data for this expert
    threshold: int = 0


@dataclass
class W2GemmOverlapArgs:
    # Number of CU used by W2 gemm
    num_sms: int
    # Same signal tensor mentioned above
    signal: torch.Tensor
    # Same as the wait_even in CombineOverlapArgs
    start_event: torch.cuda.Event


def dequant_fp8(
    expert_x_fp8: torch.Tensor, expert_x_scales: torch.Tensor
) -> torch.Tensor:
    """
    Return dequantized tensor in fp32
    """
    # TODO (varun) : Optimize leverage num_tokens_per_expert counts
    assert expert_x_fp8.is_contiguous()
    expert_x_scales = expert_x_scales.contiguous()
    num_experts = expert_x_fp8.size(0)

    expert_x_fp32 = expert_x_fp8.to(torch.float32).view(
        num_experts, -1, DEEPEP_QUANT_BLOCK_SIZE
    )
    expert_x_scales = expert_x_scales.view(num_experts, -1, 1)
    return (expert_x_fp32 * expert_x_scales).view(expert_x_fp8.size())


LOG_PREFIX = "[FP4_DISP_DBG]"


class DeepEPLLPrepareAndFinalize(mk.FusedMoEPrepareAndFinalize):
    """
    Prepare/Finalize using DeepEP low-latency kernels.
    """

    # DeepEP low-latency kernels are compiled only for certain
    # specific hidden sizes.
    # NOTE: Keep this list sorted, maybe_roundup_layer_hidden_size depends
    # on it.
    SUPPORTED_HIDDEN_SIZES = [2048, 2560, 3072, 4096, 5120, 6144, 7168, 8192]

    @staticmethod
    def maybe_roundup_layer_hidden_size(hidden_size: int) -> int:
        # Round up hidden size to the closest supported hidden size.
        _supported_hs = DeepEPLLPrepareAndFinalize.SUPPORTED_HIDDEN_SIZES
        # Check sorted
        num_supported_hs = len(_supported_hs)
        assert all(
            [
                _supported_hs[i] < _supported_hs[i + 1]
                for i in range(num_supported_hs - 1)
            ]
        )

        for x in _supported_hs:
            if x >= hidden_size:
                return x

        raise ValueError(
            f"Hidden Size {hidden_size} is greater than the "
            f"maximum supported hidden size {_supported_hs[-1]}"
        )

    def __init__(
        self,
        buffer: deep_ep.Buffer,
        max_tokens_per_rank: int,
        num_dispatchers: int,
        use_fp8_dispatch: bool = False,
        global_to_physical: torch.Tensor | None = None,
        physical_to_global: torch.Tensor | None = None,
        local_expert_global_ids: torch.Tensor | None = None,
    ):
        super().__init__()

        self.buffer = buffer
        self.max_tokens_per_rank = max_tokens_per_rank
        self.use_fp8_dispatch = use_fp8_dispatch
        # The dispatch function returns a handle that the combine function
        # requires. We store the handle here so it is available to the
        # combine function.
        self.handles: list[tuple | None] = [None, None]
        self.num_dispatchers_ = num_dispatchers

        topk_indices_dtype = self.topk_indices_dtype()

        def _maybe_cast(tensor: torch.Tensor | None) -> torch.Tensor | None:
            if tensor is None or topk_indices_dtype is None:
                return tensor
            return tensor.to(dtype=topk_indices_dtype)

        self.global_to_physical = _maybe_cast(global_to_physical)
        self.physical_to_global = _maybe_cast(physical_to_global)
        self.local_expert_global_ids = _maybe_cast(local_expert_global_ids)

        # We don't have enough information to determine if we should dispatch
        # activation scales in a packed ue8m0 format during object construction
        # time. This setting is handled by post_init_setup.
        self.use_ue8m0_dispatch = False

    def post_init_setup(self, fused_experts: mk.FusedMoEPermuteExpertsUnpermute):
        if not fused_experts.supports_packed_ue8m0_act_scales():
            # Early exit.
            return

        if self.use_fp8_dispatch:
            logger.debug_once(
                "Update DeepEPLLPrepareFinalize to do packed ue8m0 scales dispatch."
            )
            self.use_ue8m0_dispatch = True
        else:
            logger.warning_once(
                "DeepEPLLPrepareAndFinalize is setup to dispatch raw/unquantized "
                f"activations despite ({fused_experts.__class__.__name__}) being able "
                "to support quantized activations.",
                scope="local",
            )

    def num_dispatchers(self) -> int:
        return self.num_dispatchers_

    def output_is_reduced(self) -> bool:
        return True

    @property
    def activation_format(self) -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.BatchedExperts

    def max_num_tokens_per_rank(self) -> int | None:
        return self.max_tokens_per_rank

    def topk_indices_dtype(self) -> torch.dtype | None:
        return torch.int64

    def _map_global_to_physical_ids(self, topk_ids: torch.Tensor) -> torch.Tensor:
        if self.global_to_physical is None:
            return topk_ids
        return self.global_to_physical[topk_ids]

    def _map_local_to_global_ids(self, expert_topk_ids: torch.Tensor) -> torch.Tensor:
        if self.local_expert_global_ids is None:
            return expert_topk_ids
        return self.local_expert_global_ids[expert_topk_ids]

    def _do_quant(
        self,
        x: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        a1_dtype: torch.dtype,
        quant_config: FusedMoEQuantConfig,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.use_fp8_dispatch:
            block_k = (
                quant_config.block_shape[1]
                if quant_config.block_shape is not None
                else None
            )
            if block_k == DEEPEP_QUANT_BLOCK_SIZE:
                # DeepEP kernels did the quantization for us.
                x, x_scales = x
                return x, x_scales

            # Dequant to get back the tokens in the datatype we dispatched in.
            x_fp8, x_scales = x
            x = dequant_fp8(x_fp8, x_scales).to(dtype=a1_dtype)

        assert isinstance(x, (torch.Tensor, tuple))
        q_dtype = quant_config.quant_dtype

        if q_dtype == "nvfp4" and envs.VLLM_DEEPEPLL_NVFP4_DISPATCH:
            logger.info_once(
                "Since VLLM_DEEPEPLL_NVFP4_DISPATCH==1, make sure "
                "using the hybrid-ep branch of DeepEP"
                "(https://github.com/deepseek-ai/DeepEP/tree/hybrid-ep)"
            )
            assert isinstance(x, tuple)
            x_scales = x[1]
            x = x[0].permute(2, 0, 1)
            num_experts, max_tokens, hidden_dim_by_2 = x.shape
            hidden_dim = hidden_dim_by_2 * 2
            assert envs.VLLM_FLASHINFER_MOE_BACKEND == "masked_gemm"
            logger.info_once(
                "Quantization is fused with DeepEP nvfp4 dispatch for "
                "FlashInfer CUTEDSL as VLLM_DEEPEPLL_NVFP4_DISPATCH==1"
            )
            logger.info(
                f"{LOG_PREFIX} _do_quant NVFP4_DISPATCH path: "
                f"x.shape={x.shape}, x.dtype={x.dtype}, "
                f"x_scales.shape={x_scales.shape}, x_scales.dtype={x_scales.dtype}, "
                f"num_experts={num_experts}, max_tokens={max_tokens}, "
                f"hidden_dim={hidden_dim}"
            )
        else:
            if q_dtype == "nvfp4":
                q_dtype = None
                logger.info_once(
                    "Using DeepEP bfloat16 dispatch for FlashInfer CUTEDSL as "
                    "VLLM_DEEPEPLL_NVFP4_DISPATCH==0"
                )
            assert isinstance(x, torch.Tensor)
            num_experts, max_tokens, hidden_dim = x.size()

            # TODO (varun): Optimization - Use a batched version of quant
            x = x.view((-1, hidden_dim))
            logger.info(
                f"{LOG_PREFIX} no fp4 dispatch: calling moe_kernel_quantize_input before: {x.dtype=}"
            )
            x, x_scales = moe_kernel_quantize_input(
                x,
                quant_config.a1_scale,
                q_dtype,
                quant_config.per_act_token_quant,
                quant_config.block_shape,
            )
            x = x.view((num_experts, -1, hidden_dim))
            logger.info(
                f"{LOG_PREFIX} no fp4 dispatch: calling moe_kernel_quantize_input after: {x.dtype=}"
            )

        if q_dtype is not None and q_dtype != "nvfp4":
            assert x_scales is not None
            x_scales = normalize_batched_scales_shape(x_scales, num_experts)

        return x, x_scales

    def supports_async(self) -> bool:
        return True

    def prepare_async(
        self,
        a1: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        local_num_experts: int,
        expert_map: torch.Tensor | None,
        apply_router_weight_on_input: bool,
        quant_config: FusedMoEQuantConfig,
    ) -> tuple[Callable, mk.ReceiverType]:
        hidden_size = a1.size(1)
        assert hidden_size in self.SUPPORTED_HIDDEN_SIZES, (
            f"Hidden Size {hidden_size} not in supported list of hidden sizes"
            f"{self.SUPPORTED_HIDDEN_SIZES}"
        )

        a2a_idx = dbo_current_ubatch_id()

        if self.use_fp8_dispatch:
            assert hidden_size % 128 == 0, (
                "DeepEP kernels quantize the inputs in blocks of shape 128"
            )

        use_nvfp4 = False
        nvfp4_dispatch = (
            quant_config.quant_dtype == "nvfp4" and envs.VLLM_DEEPEPLL_NVFP4_DISPATCH
        )
        if nvfp4_dispatch:
            use_nvfp4 = True
        qc_a1_gscale_or_scale = (
            quant_config.a1_gscale if nvfp4_dispatch else quant_config.a1_scale
        )
        has_per_token_scales = (
            qc_a1_gscale_or_scale.numel() != 1
            if qc_a1_gscale_or_scale is not None
            else (
                quant_config.a2_scale.numel() != 1
                if quant_config.a2_scale is not None
                else False
            )
        )
        logger.info(
            f"{LOG_PREFIX} prepare_async: a1.shape={a1.shape}, a1.dtype={a1.dtype}, "
            f"hidden_size={hidden_size}, num_experts={num_experts}, "
            f"local_num_experts={local_num_experts}, "
            f"nvfp4_dispatch={nvfp4_dispatch}, use_nvfp4={use_nvfp4}, "
            f"quant_dtype={quant_config.quant_dtype}, "
            f"a1_gscale={quant_config.a1_gscale}, a1_scale={quant_config.a1_scale}, "
            f"max_tokens_per_rank={self.max_tokens_per_rank}"
        )

        # Log input tensor stats for MoE before dispatch
        a1_float = a1.float()
        a1_flat = a1_float.flatten()
        num_samples = min(10, len(a1_flat))
        sample_indices = [
            int(i * len(a1_flat) / num_samples) for i in range(num_samples)
        ]
        samples = [a1_flat[idx].item() for idx in sample_indices]
        has_nan = torch.isnan(a1_float).any().item()
        has_inf = torch.isinf(a1_float).any().item()
        nan_count = torch.isnan(a1_float).sum().item()
        inf_count = torch.isinf(a1_float).sum().item()
        nonzero_count = (a1_float != 0).sum().item()
        logger.info(
            f"{LOG_PREFIX} before_dispatch: "
            f"shape={a1.shape}, dtype={a1.dtype}, "
            f"min={a1_float.min().item():.6f}, max={a1_float.max().item():.6f}, "
            f"mean={a1_float.mean().item():.6f}, std={a1_float.std().item():.6f}, "
            f"has_nan={has_nan}, has_inf={has_inf}, nan_count={nan_count}, inf_count={inf_count}, "
            f"nonzero_count={nonzero_count}, total={a1_flat.numel()}, "
            f"samples={[f'{s:.6f}' for s in samples]}, "
            f"nvfp4_dispatch={nvfp4_dispatch}, "
            f"a1_gscale_shape={quant_config.a1_gscale.shape if quant_config.a1_gscale is not None else None}, "
            f"a1_scale_shape={quant_config.a1_scale.shape if quant_config.a1_scale is not None else None}, "
            f"a1_gscale={quant_config.a1_gscale[:10].tolist() if quant_config.a1_gscale is not None and len(quant_config.a1_gscale) >= 10 else (quant_config.a1_gscale.tolist() if quant_config.a1_gscale is not None else None)}"
        )

        if not use_nvfp4:
            assert not has_per_token_scales, (
                "low_latency kernels doesn't support dispatching per-token scales"
            )

        if apply_router_weight_on_input:
            topk = topk_ids.size(1)
            # TODO: this only works for topK=1, will need to update for topK>1
            assert topk == 1, (
                "apply_router_weight_on_input is only implemented for topk=1"
            )
            a1 = a1 * topk_weights.to(a1.dtype)

        # Dispatch
        dispatch_topk_ids = self._map_global_to_physical_ids(topk_ids)
        expert_x, expert_num_tokens, handle, _, hook = self.buffer.low_latency_dispatch(
            a1,
            dispatch_topk_ids,
            self.max_tokens_per_rank,
            num_experts,
            use_fp8=self.use_fp8_dispatch,
            **(dict(use_nvfp4=True) if use_nvfp4 else dict()),
            **(
                dict(x_global_scale=qc_a1_gscale_or_scale)
                if qc_a1_gscale_or_scale is not None
                else dict()
            ),
            async_finish=False,
            return_recv_hook=True,
        )
        self.handles[a2a_idx] = handle

        # We need to pass w2_gemm_overlap_args to moe implementation,
        # so return it as an output paramter
        w2_gemm_overlap_args = self._create_sbo_args(local_num_experts, a1.device)
        return (
            hook,
            lambda: self._receiver(
                expert_x,
                expert_num_tokens,
                quant_config.a1_scale,
                a1.dtype,
                quant_config,
            ),
            w2_gemm_overlap_args,
        )

    def _receiver(
        self,
        expert_x: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        expert_num_tokens: torch.Tensor,
        a1_scale: torch.Tensor | None,
        a1_dtype: torch.dtype,
        quant_config: FusedMoEQuantConfig,
    ) -> mk.PrepareResultType:
        # Log input to _receiver (after dispatch, before _do_quant)
        if isinstance(expert_x, tuple):
            logger.info(
                f"{LOG_PREFIX} _receiver input (tuple): "
                f"expert_x[0].shape={expert_x[0].shape}, expert_x[0].dtype={expert_x[0].dtype}, "
                f"expert_x[1].shape={expert_x[1].shape}, expert_x[1].dtype={expert_x[1].dtype}, "
                f"dispatch_scale_shape={expert_x[1].shape}, "
                f"quant_dtype={quant_config.quant_dtype}"
            )
        else:
            logger.info(
                f"{LOG_PREFIX} _receiver input: "
                f"expert_x.shape={expert_x.shape}, expert_x.dtype={expert_x.dtype}, "
                f"dispatch_scale_shape=None, "
                f"quant_dtype={quant_config.quant_dtype}"
            )

        expert_x, expert_x_scale = self._do_quant(expert_x, a1_dtype, quant_config)

        # Log output after _do_quant with detailed tensor stats and samples
        if expert_x_scale is not None:
            # Quantized path - log both data and scales
            ex_flat = expert_x.flatten().float()
            scale_flat = expert_x_scale.flatten().float()
            num_samples = min(10, len(ex_flat))
            sample_indices = [
                int(i * len(ex_flat) / num_samples) for i in range(num_samples)
            ]
            data_samples = [ex_flat[idx].item() for idx in sample_indices]
            num_samples_scale = min(10, len(scale_flat))
            sample_indices_scale = [
                int(i * len(scale_flat) / num_samples_scale)
                for i in range(num_samples_scale)
            ]
            scale_samples = [scale_flat[idx].item() for idx in sample_indices_scale]
            # Stats for data
            data_nan_count = torch.isnan(ex_flat).sum().item()
            data_inf_count = torch.isinf(ex_flat).sum().item()
            data_nonzero = (ex_flat != 0).sum().item()
            data_min = ex_flat.min().item()
            data_max = ex_flat.max().item()
            data_mean = ex_flat.mean().item()
            data_std = ex_flat.std().item()
            # Stats for scales
            scale_nan_count = torch.isnan(scale_flat).sum().item()
            scale_inf_count = torch.isinf(scale_flat).sum().item()
            scale_nonzero = (scale_flat != 0).sum().item()
            scale_negative = (scale_flat < 0).sum().item()
            logger.info(
                f"{LOG_PREFIX} after_dispatch (quantized): "
                f"data_shape={expert_x.shape}, data_dtype={expert_x.dtype}, "
                f"scale_shape={expert_x_scale.shape}, scale_dtype={expert_x_scale.dtype}, "
                f"data_min={data_min:.6f}, data_max={data_max:.6f}, "
                f"data_mean={data_mean:.6f}, data_std={data_std:.6f}, "
                f"data_nonzero={data_nonzero}, data_nan_count={data_nan_count}, "
                f"data_inf_count={data_inf_count}, data_total={ex_flat.numel()}, "
                f"data_samples={[f'{s:.6f}' for s in data_samples]}, "
                f"scale_min={scale_flat.min().item():.6f}, "
                f"scale_max={scale_flat.max().item():.6f}, "
                f"scale_mean={scale_flat.mean().item():.6f}, "
                f"scale_std={scale_flat.std().item():.6f}, "
                f"scale_nan_count={scale_nan_count}, scale_inf_count={scale_inf_count}, "
                f"scale_nonzero={scale_nonzero}, scale_negative={scale_negative}, "
                f"scale_total={scale_flat.numel()}, "
                f"scale_samples={[f'{s:.6f}' for s in scale_samples]}"
            )

            # Debug: Investigate NaN and negative scales
            if scale_nan_count > 0 or scale_negative > 0:
                logger.info(
                    f"{LOG_PREFIX} scale_debug: expert_num_tokens={expert_num_tokens.tolist()}"
                )
                # Analyze NaN/negative distribution per expert
                # expert_x_scale shape is [num_experts, ...] after reshape or swizzled
                num_experts = expert_x_scale.shape[0]
                for exp_idx in range(num_experts):
                    exp_scale = expert_x_scale[exp_idx].flatten().float()
                    exp_nan = torch.isnan(exp_scale).sum().item()
                    exp_neg = (exp_scale < 0).sum().item()
                    exp_total = exp_scale.numel()
                    exp_tokens = expert_num_tokens[exp_idx].item()
                    if exp_nan > 0 or exp_neg > 0:
                        # Get valid (non-NaN) stats
                        valid_mask = ~torch.isnan(exp_scale)
                        valid_scales = exp_scale[valid_mask]
                        valid_min = (
                            valid_scales.min().item()
                            if valid_scales.numel() > 0
                            else float("nan")
                        )
                        valid_max = (
                            valid_scales.max().item()
                            if valid_scales.numel() > 0
                            else float("nan")
                        )
                        logger.info(
                            f"{LOG_PREFIX} scale_debug expert[{exp_idx}]: "
                            f"tokens={exp_tokens}, nan={exp_nan}, neg={exp_neg}, "
                            f"total={exp_total}, valid_min={valid_min:.6f}, valid_max={valid_max:.6f}"
                        )

                # Check if NaNs are in padding regions (tokens beyond expert_num_tokens)
                # For swizzled layout [num_experts, 4, 16, 4, 112, 32], need to understand mapping
                max_tokens = expert_x.shape[1]  # max_tokens_per_rank
                logger.info(
                    f"{LOG_PREFIX} scale_debug: max_tokens_per_rank={max_tokens}, "
                    f"total_tokens_received={expert_num_tokens.sum().item()}, "
                    f"scale_layout={expert_x_scale.shape}"
                )
        else:
            # Non-quantized path
            ex_float = expert_x.float()
            ex_flat = ex_float.flatten()
            num_samples = min(10, len(ex_flat))
            sample_indices = [
                int(i * len(ex_flat) / num_samples) for i in range(num_samples)
            ]
            samples = [ex_flat[idx].item() for idx in sample_indices]
            has_nan = torch.isnan(ex_float).any().item()
            has_inf = torch.isinf(ex_float).any().item()
            nan_count = torch.isnan(ex_float).sum().item()
            inf_count = torch.isinf(ex_float).sum().item()
            nonzero_count = (ex_float != 0).sum().item()
            logger.info(
                f"{LOG_PREFIX} after_dispatch (tensor): "
                f"shape={expert_x.shape}, dtype={expert_x.dtype}, "
                f"min={ex_float.min().item():.6f}, max={ex_float.max().item():.6f}, "
                f"mean={ex_float.mean().item():.6f}, std={ex_float.std().item():.6f}, "
                f"has_nan={has_nan}, has_inf={has_inf}, "
                f"nan_count={nan_count}, inf_count={inf_count}, "
                f"nonzero_count={nonzero_count}, total={ex_flat.numel()}, "
                f"samples={[f'{s:.6f}' for s in samples]}"
            )

        expert_tokens_meta = mk.ExpertTokensMetadata(
            expert_num_tokens=expert_num_tokens, expert_num_tokens_cpu=None
        )

        return expert_x, expert_x_scale, expert_tokens_meta, None, None

    def prepare(
        self,
        a1: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        local_num_experts: int,
        expert_map: torch.Tensor | None,
        apply_router_weight_on_input: bool,
        quant_config: FusedMoEQuantConfig,
    ) -> mk.PrepareResultType:
        hook, receiver = self.prepare_async(
            a1,
            topk_weights,
            topk_ids,
            num_experts,
            expert_map,
            apply_router_weight_on_input,
            quant_config,
        )
        hook()
        return receiver()

    def _finalize(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        weight_and_reduce_impl: mk.TopKWeightAndReduce,
        do_async: bool,
    ) -> tuple[Callable, Callable]:
        assert isinstance(weight_and_reduce_impl, TopKWeightAndReduceDelegate), (
            "Weight application and reduction happens in the combine kernel."
        )

        # Log MoE output stats before combine with samples
        fused_float = fused_expert_output.float()
        fused_flat = fused_float.flatten()
        num_samples = min(10, len(fused_flat))
        sample_indices = [
            int(i * len(fused_flat) / num_samples) for i in range(num_samples)
        ]
        samples = [fused_flat[idx].item() for idx in sample_indices]
        has_nan = torch.isnan(fused_float).any().item()
        has_inf = torch.isinf(fused_float).any().item()
        nan_count = torch.isnan(fused_float).sum().item()
        inf_count = torch.isinf(fused_float).sum().item()
        nonzero_count = (fused_float != 0).sum().item()
        logger.info(
            f"{LOG_PREFIX} before_combine: "
            f"shape={fused_expert_output.shape}, dtype={fused_expert_output.dtype}, "
            f"min={fused_float.min().item():.6f}, max={fused_float.max().item():.6f}, "
            f"mean={fused_float.mean().item():.6f}, std={fused_float.std().item():.6f}, "
            f"has_nan={has_nan}, has_inf={has_inf}, nan_count={nan_count}, inf_count={inf_count}, "
            f"nonzero_count={nonzero_count}, total={fused_flat.numel()}, "
            f"samples={[f'{s:.6f}' for s in samples]}"
        )

        a2a_idx = dbo_current_ubatch_id()
        do_recv_hook = dbo_enabled() or do_async
        handle = self.handles[a2a_idx]
        assert handle is not None

        combine_topk_weights = topk_weights
        if apply_router_weight_on_input:
            # weights have already been applied.
            combine_topk_weights = torch.ones_like(topk_weights)

        combine_topk_ids = self._map_global_to_physical_ids(topk_ids)
        # TODO (varun) : Enable zero copy mode
        dbo_maybe_run_recv_hook()
        ctx = nullcontext()

        if self.combine_overlap_args is not None:
            # For SBO, we need to wait for compute stream
            # to have completed signal tensor allocation
            self.combine_overlap_args.stream.wait_event(
                self.combine_overlap_args.wait_event
            )
            # And we launch ll combine phase 1 in a separate stream
            # for overlaping
            ctx = torch.cuda.stream(self.combine_overlap_args.stream)
        with ctx:
            _, _, recv_hook = self.buffer.low_latency_combine(
                fused_expert_output,
                topk_ids,
                combine_topk_weights,
                handle,
                async_finish=False,
                zero_copy=False,
                return_recv_hook=do_recv_hook,
                out=output,
                **(
                    dict(
                        overlap=True,
                        src_signals=self.combine_overlap_args.signal,
                        src_signal_expect_value=self.combine_overlap_args.threshold,
                    )
                    if self.combine_overlap_args is not None
                    else {}
                ),
            )

        # Log after combine with tensor stats and samples
        out_float = output.float()
        out_flat = out_float.flatten()
        num_samples = min(10, len(out_flat))
        sample_indices = [
            int(i * len(out_flat) / num_samples) for i in range(num_samples)
        ]
        samples = [out_flat[idx].item() for idx in sample_indices]
        has_nan = torch.isnan(out_float).any().item()
        has_inf = torch.isinf(out_float).any().item()
        nan_count = torch.isnan(out_float).sum().item()
        inf_count = torch.isinf(out_float).sum().item()
        nonzero_count = (out_float != 0).sum().item()
        logger.info(
            f"{LOG_PREFIX} after_combine: "
            f"shape={output.shape}, dtype={output.dtype}, "
            f"min={out_float.min().item():.6f}, max={out_float.max().item():.6f}, "
            f"mean={out_float.mean().item():.6f}, std={out_float.std().item():.6f}, "
            f"has_nan={has_nan}, has_inf={has_inf}, nan_count={nan_count}, inf_count={inf_count}, "
            f"nonzero_count={nonzero_count}, total={out_flat.numel()}, "
            f"samples={[f'{s:.6f}' for s in samples]}"
        )

        return recv_hook, lambda: self._sbo_wait_stream()

    def finalize_async(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        weight_and_reduce_impl: mk.TopKWeightAndReduce,
    ) -> tuple[Callable, Callable]:
        return self._finalize(
            output,
            fused_expert_output,
            topk_weights,
            topk_ids,
            apply_router_weight_on_input,
            weight_and_reduce_impl,
            do_async=True,
        )

    def finalize(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        weight_and_reduce_impl: mk.TopKWeightAndReduce,
    ) -> None:
        self._finalize(
            output,
            fused_expert_output,
            topk_weights,
            topk_ids,
            apply_router_weight_on_input,
            weight_and_reduce_impl,
            do_async=False,
        )

    def _create_sbo_args(
        self, local_num_experts: int, device: torch.device
    ) -> W2GemmOverlapArgs:
        w2_gemm_overlap_args = None
        self.combine_overlap_args = None
        if envs.VLLM_EP_USE_SBO:
            total_num_sms = torch.cuda.get_device_properties(
                device="cuda"
            ).multi_processor_count
            communicate_num_sms = 56
            compute_num_sms = total_num_sms - communicate_num_sms

            combine_wait_event = torch.cuda.Event()
            combine_overlap_args = CombineOverlapArgs(
                num_sms=communicate_num_sms,
                stream=alt_stream,
                wait_event=combine_wait_event,
            )

            combine_signal = torch.zeros(
                local_num_experts, dtype=torch.uint32, device=device
            )

            w2_gemm_overlap_args = W2GemmOverlapArgs(
                signal=combine_signal,
                start_event=combine_wait_event,
                num_sms=compute_num_sms,
            )
            combine_overlap_args.signal = combine_signal
            combine_overlap_args.threshold = compute_num_sms
            self.combine_overlap_args = combine_overlap_args
        return w2_gemm_overlap_args

    def _sbo_wait_stream(self) -> None:
        # When SBO enabled, ll combine phase 2 is still launched
        # on the main compute stream, but we need to wait for
        # ll combine 1 to complete
        if self.combine_overlap_args is not None:
            torch.cuda.current_stream().wait_stream(self.combine_overlap_args.stream)
