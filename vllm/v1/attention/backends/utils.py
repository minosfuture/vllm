# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import abc
import enum
import functools
from abc import abstractmethod
from dataclasses import dataclass, make_dataclass
from typing import Any, ClassVar, Generic, Optional, TYPE_CHECKING, TypeVar

import numpy as np
import torch

from vllm.config import get_layers_from_vllm_config, VllmConfig
from vllm.utils import cdiv

if TYPE_CHECKING:
    from vllm.attention.backends.abstract import AttentionImpl
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.worker.gpu_input_batch import InputBatch

import vllm.envs as envs
from vllm.attention.backends.abstract import AttentionBackend
from vllm.attention.layer import Attention
from vllm.distributed.kv_transfer.kv_connector.utils import (
    get_kv_connector_cache_layout,
)
from vllm.logger import init_logger
from vllm.v1.kv_cache_interface import AttentionSpec

logger = init_logger(__name__)
_KV_CACHE_LAYOUT_OVERRIDE = None


@dataclass
class CommonAttentionMetadata:
    """
    Per-batch attention metadata, shared across layers and backends.
    AttentionMetadataBuilder instances use it to construct per-layer metadata.

    For many of the tensors we keep both GPU and CPU versions.
    """

    query_start_loc: torch.Tensor
    query_start_loc_cpu: torch.Tensor
    """(batch_size + 1,), the start location of each request in query Tensor"""

    seq_lens: torch.Tensor
    seq_lens_cpu: torch.Tensor
    """(batch_size,), the length of each request including both computed tokens
    and newly scheduled tokens"""

    num_computed_tokens_cpu: torch.Tensor
    """(batch_size,), the number of computed tokens for each request"""

    num_reqs: int
    """Number of requests"""
    num_actual_tokens: int
    """Total number of tokens in batch"""
    max_query_len: int
    """Longest query in batch"""
    max_seq_len: int
    """Longest context length in batch"""

    block_table_tensor: torch.Tensor
    slot_mapping: torch.Tensor

    causal: bool = True


@dataclass
class UbatchSlice:
    request_slice: slice
    token_slice: slice
    # Additional fields to support prefill operations
    compute_complexity: Optional[float] = (
        None  # Estimated compute complexity for load balancing
    )
    query_lens: Optional[torch.Tensor] = (
        None  # Query lengths for each request in this ubatch
    )
    has_prefill: bool = False  # Whether this ubatch contains prefill operations
    max_query_len: int = 1  # Maximum query length in this ubatch

    # New fields to support non-consecutive request/token indices
    request_indices: Optional[torch.Tensor] = (
        None  # Specific request indices (alternative to slice)
    )
    token_indices: Optional[torch.Tensor] = (
        None  # Specific token indices (alternative to slice)
    )


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
    compute_complexities: torch.Tensor  # Estimated compute complexity per request


def slice_query_start_locs(
    query_start_loc: torch.Tensor,
    request_slice: slice,
) -> torch.Tensor:
    """
    Creates a new query_start_loc that corresponds to the requests in
    request_slice.

    Note: This function creates a new tensor to hold the new query_start_locs.
    This will break cudagraph compatibility.
    """
    return (
        query_start_loc[request_slice.start : request_slice.stop + 1]
        - query_start_loc[request_slice.start]
    )


def _make_metadata_with_slice(
    ubatch_slice: UbatchSlice, attn_metadata: CommonAttentionMetadata
) -> CommonAttentionMetadata:
    """
    This function creates a new CommonAttentionMetadata that corresponds to
    the requests included in ubatch_slice
    """
    logger.debug(
        f"[UBatch Slice] Creating metadata slice - request_slice: {ubatch_slice.request_slice}, "
        f"token_slice: {ubatch_slice.token_slice}, has_prefill: {ubatch_slice.has_prefill}"
    )

    request_slice = ubatch_slice.request_slice
    token_slice = ubatch_slice.token_slice

    query_start_loc = slice_query_start_locs(
        attn_metadata.query_start_loc, request_slice
    )
    assert len(query_start_loc) >= 2, (
        f"query_start_loc must have at least 2 elements, " f"got {len(query_start_loc)}"
    )
    query_start_loc_cpu = slice_query_start_locs(
        attn_metadata.query_start_loc_cpu, request_slice
    )

    seq_lens = attn_metadata.seq_lens[request_slice]
    seq_lens_cpu = attn_metadata.seq_lens_cpu[request_slice]
    max_seq_len = int(seq_lens_cpu.max())
    num_computed_tokens_cpu = attn_metadata.num_computed_tokens_cpu[request_slice]

    num_requests = request_slice.stop - request_slice.start
    num_actual_tokens = token_slice.stop - token_slice.start
    max_query_len = int(
        torch.max(torch.abs(query_start_loc_cpu[1:] - query_start_loc_cpu[:-1])).item()
    )

    block_table_tensor = attn_metadata.block_table_tensor[request_slice]
    slot_mapping = attn_metadata.slot_mapping[token_slice]

    logger.debug(
        f"[UBatch Slice] Created metadata - num_requests: {num_requests}, "
        f"num_actual_tokens: {num_actual_tokens}, max_query_len: {max_query_len}, "
        f"max_seq_len: {max_seq_len}"
    )

    return CommonAttentionMetadata(
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc_cpu,
        seq_lens=seq_lens,
        seq_lens_cpu=seq_lens_cpu,
        num_computed_tokens_cpu=num_computed_tokens_cpu,
        num_reqs=num_requests,
        num_actual_tokens=num_actual_tokens,
        max_query_len=max_query_len,
        max_seq_len=max_seq_len,
        block_table_tensor=block_table_tensor,
        slot_mapping=slot_mapping,
    )


def split_attn_metadata(
    ubatch_slices: list[UbatchSlice],
    common_attn_metadata: CommonAttentionMetadata,
) -> list[CommonAttentionMetadata]:
    """
    Creates a new CommonAttentionMetadata instance that corresponds to the
    requests for each UbatchSlice in ubatch_slices.

    Note: This function does not modify common_attn_metadata
    """
    logger.debug(
        f"[UBatch Split] Splitting attention metadata into {len(ubatch_slices)} ubatches. "
        f"Original batch - num_reqs: {common_attn_metadata.num_reqs}, "
        f"num_tokens: {common_attn_metadata.num_actual_tokens}, "
        f"max_query_len: {common_attn_metadata.max_query_len}, "
        f"query_start_loc: {common_attn_metadata.query_start_loc_cpu}"
    )

    results = []
    for i, ubatch_slice in enumerate(ubatch_slices):
        logger.debug(f"[UBatch Split] Processing ubatch {i}/{len(ubatch_slices)}")
        result = _make_metadata_with_slice(ubatch_slice, common_attn_metadata)
        # Update the metadata with prefill-aware information
        if ubatch_slice.has_prefill:
            # For prefill ubatches, ensure we have the correct max_query_len
            result.max_query_len = ubatch_slice.max_query_len
            logger.debug(
                f"[UBatch Split] Ubatch {i} has prefill, max_query_len: {ubatch_slice.max_query_len}"
            )

        results.append(result)

    logger.debug(
        f"[UBatch Split] Successfully split metadata into {len(results)} ubatches"
    )
    return results


M = TypeVar("M")


class AttentionCGSupport(enum.Enum):
    """Constants for the cudagraph support of the attention backend
    Here we do not consider the cascade attention, as currently
    it is never cudagraph supported."""

    ALWAYS = 3
    """Cudagraph always supported; supports mixed-prefill-decode"""
    UNIFORM_BATCH = 2
    """Cudagraph supported for batches the only contain query lengths that are
    the same, this can be used for spec-decode
        i.e. "decodes" are 1 + num_speculative_tokens"""
    UNIFORM_SINGLE_TOKEN_DECODE = 1
    """Cudagraph supported for batches the only contain query_len==1 decodes"""
    NEVER = 0
    """NO cudagraph support"""


class AttentionMetadataBuilder(abc.ABC, Generic[M]):
    # Does this backend/builder support CUDA Graphs for attention (default: no).
    cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.NEVER
    # Does this backend/builder reorder the batch?
    # If not, set this to None. Otherwise set it to the query
    # length that will be pulled into the front of the batch.
    reorder_batch_threshold: ClassVar[Optional[int]] = None

    @abstractmethod
    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        self.kv_cache_spec = kv_cache_spec

    @abstractmethod
    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> M:
        """
        Central method that builds attention metadata.
        Some builders (MLA) require reorder_batch to be called prior to build.

        Args:
            common_prefix_len: The length of the common prefix of the batch.
            common_attn_metadata: The common attention metadata.
            fast_build: The meta-data will prioritize speed of building over
                then speed at execution. Can be used for spec-decode where the
                result of a build call may only be used for few layers/iters.
        """
        raise NotImplementedError

    def build_for_cudagraph_capture(
        self, common_attn_metadata: CommonAttentionMetadata
    ) -> M:
        """
        Build attention metadata for CUDA graph capture. Uses build by default.
        Subclasses that override this method should call self.build or
        super().build_for_cudagraph_capture.
        """
        return self.build(
            common_prefix_len=0, common_attn_metadata=common_attn_metadata
        )

    def build_for_drafting(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        draft_index: int,
    ) -> M:
        """
        Build attention metadata for draft model. Uses build by default.

        Args:
            common_attn_metadata: The common attention metadata.
            draft_index: The index of the current draft operation.
                When speculating a chain of tokens, this index refers to the
                draft attempt for the i-th token.
                For tree-based attention, this index instead refers to the
                draft attempt for the i-th level in the tree of tokens.
        """
        return self.build(
            common_prefix_len=0,
            common_attn_metadata=common_attn_metadata,
            fast_build=True,
        )

    def use_cascade_attention(
        self,
        common_prefix_len: int,
        query_lens: np.ndarray,
        num_query_heads: int,
        num_kv_heads: int,
        use_alibi: bool,
        use_sliding_window: bool,
        use_local_attention: bool,
        num_sms: int,
    ) -> bool:
        return False


@functools.lru_cache
def get_kv_cache_layout():
    # Format specified by the code.
    global _KV_CACHE_LAYOUT_OVERRIDE

    if _KV_CACHE_LAYOUT_OVERRIDE is not None:
        cache_layout = _KV_CACHE_LAYOUT_OVERRIDE
        logger.info_once(
            "`_KV_CACHE_LAYOUT_OVERRIDE` variable detected. "
            "Setting KV cache layout to %s.",
            cache_layout,
        )
        return cache_layout

    # Format specified by the user.
    cache_layout = envs.VLLM_KV_CACHE_LAYOUT
    # When neither the user nor the override specified a layout, get default
    if cache_layout is None:
        cache_layout = get_kv_connector_cache_layout()
    else:
        logger.info_once(
            "`VLLM_KV_CACHE_LAYOUT` environment variable "
            "detected. Setting KV cache layout to %s.",
            cache_layout,
        )
    return cache_layout


def set_kv_cache_layout(cache_layout: str):
    global _KV_CACHE_LAYOUT_OVERRIDE
    _KV_CACHE_LAYOUT_OVERRIDE = cache_layout


@dataclass
class PerLayerParameters:
    """
    Currently, FlashInfer backend only support models in which all layers share
    the same values for the following hyperparameters. Should not be used for
    trtllm-gen backend since it supports different values for the following
    hyperparameters.
    """

    window_left: int
    logits_soft_cap: Optional[float]
    sm_scale: float
    has_sinks: bool = False


def get_per_layer_parameters(
    vllm_config: VllmConfig, layer_names: list[str], cls_: type["AttentionImpl"]
) -> dict[str, PerLayerParameters]:
    """
    Scan layers in `layer_names` and determine some hyperparameters
    to use during `plan`.
    """

    layers = get_layers_from_vllm_config(vllm_config, Attention, layer_names)
    per_layer_params: dict[str, PerLayerParameters] = {}

    for key, layer in layers.items():
        impl = layer.impl
        assert isinstance(impl, cls_)

        # Infer hyperparameters from the attention layer
        window_size = getattr(impl, "sliding_window", None)
        window_left = window_size[0] if window_size is not None else -1
        logits_soft_cap = getattr(impl, "logits_soft_cap", None)
        sm_scale = impl.scale
        has_sinks = getattr(impl, "sinks", None) is not None

        per_layer_params[key] = PerLayerParameters(
            window_left, logits_soft_cap, sm_scale, has_sinks
        )

    return per_layer_params


def infer_global_hyperparameters(
    per_layer_params: dict[str, PerLayerParameters]
) -> PerLayerParameters:
    """
    Currently, FlashInfer backend other than trtllm-gen
    only support models in which all layers share
    the same values for the following hyperparameters:
    - `window_left`
    - `logits_soft_cap`
    - `sm_scale`

    So this function asserts that all layers share the same values for these
    hyperparameters and returns the global values.
    """

    assert len(per_layer_params) > 0, "No attention layers found in the model."

    param_sets = list(per_layer_params.values())
    global_params = param_sets[0]

    # trtllm attention doesn't need global hyper params so disable the check
    if not envs.VLLM_USE_TRTLLM_ATTENTION:
        for params in param_sets:
            if params.window_left != global_params.window_left:
                raise ValueError(
                    "Window left is not the same for all layers. "
                    "One potential fix is to set disable_sliding_window=True"
                )
            assert params == global_params, (
                "FlashInfer backend currently only supports models in which all"
                "layers share the same values "
                "for the following hyperparameters:"
                "`window_left`, `logits_soft_cap`, `sm_scale`."
            )

    return global_params


#
# Take in `query_start_loc_np` and `seq_lens_np` and break the sequences into
# local attention blocks, where each block is passed to the attention kernel
# as an independent local ("virtual") batch item.
#
# For example, if are performing a chunked prefill a batch of 3 sequences:
#   q_seqlens  = [4, 10, 5]
#   kv_seqlens = [6, 17, 9]
# Then normally for regular attention we would compute with an attention mask
#  for batch idx 0 (q_seqlens = 4, kv_seqlens = 6) like:
#   batch idx: 0 (q_seqlens = 4, kv_seqlens = 6)
#        k_toks >   0 1 2 3 4 5
#        q_toks v  _____________
#               0 | 1 1 1
#               1 | 1 1 1 1
#               2 | 1 1 1 1 1
#               3 | 1 1 1 1 1 1
#
# for local attention (with attn_chunk_size = 4) we would compute with an
#  attention mask like:
#   batch idx: 0  (q_seqlens = 4, kv_seqlens = 6, attn_chunk_size = 4)
#        k_toks >   0 1 2 3 4 5
#        q_toks v  _____________
#               0 | 1 1 1
#               1 | 1 1 1 1
#               2 |         1
#               3 |         1 1
#
# We can simulate this mask using standard flash-attention by breaking the
#  sequences into local ("virtual") batches, where each local batch item is a
#  local attention block, so in this case batch idx 0 would be broken up into:
#
#   local-batch idx: 0 (q_seqlens = 2, kv_seqlens = 4)  (batch 0)
#        k_toks >   0 1 2 3
#        q_toks v  _____________
#               0 | 1 1 1
#               1 | 1 1 1 1
#   local-batch idx: 1 (q_seqlens = 2, kv_seqlens = 2) (batch 0)
#        k_toks >   4 5
#        q_toks v  _____________
#               2 | 1
#               3 | 1 1
#
# e.g. if we have:
#   attn_chunk_size = 4
#   query_start_loc_np = [0, 4, 14, 19] (q_seqlens = [4, 10, 5])
# Then this function would return:
#                           __b0__  ______b1______  __b2__ < orig batch indices
#   q_seqlens_local    = [   2,  2,  1,  4,  4,  1,  4,  1]
#   cu_seqlens_q_local = [0, 4,  6, 10, 14, 18, 19, 23, 24]
#   seqlens_k_local    = [   4,  2,  4,  4,  4,  1,  4,  1]
#   block_table_local  : shape[local_virtual_batches, pages_per_local_batch]
def make_local_attention_virtual_batches(
    attn_chunk_size: int,
    common_attn_metadata: CommonAttentionMetadata,
    block_size: int = 0,
) -> CommonAttentionMetadata:
    query_start_loc_np = common_attn_metadata.query_start_loc_cpu.numpy()
    seq_lens_np = common_attn_metadata.seq_lens_cpu.numpy()
    block_table = common_attn_metadata.block_table_tensor
    device = common_attn_metadata.query_start_loc.device

    q_seqlens = query_start_loc_np[1:] - query_start_loc_np[:-1]
    actual_batch_size = seq_lens_np.shape[0]

    # Handle if we are starting in the middle of a local attention block,
    #  we assume q_seqlens > 0 (for all elements), for each batch idx we compute
    #  the number of tokens that are not in the first local attention block and
    #  then we can simply use a cdiv for the rest.
    # For example if we have:
    #   attn_chunk_size = 4
    #   q_seqlens = [4, 10, 5]
    #   k_seqlens = [6, 17, 9]
    # Then we would get:
    #   new_tokens_in_first_block = [2, 1, 4]
    #   local_blocks = [2, 4, 2]
    q_tokens_in_first_block = np.minimum(
        attn_chunk_size - ((seq_lens_np - q_seqlens) % attn_chunk_size), q_seqlens
    ).astype(np.int32)
    tokens_in_last_block = attn_chunk_size + (seq_lens_np % -attn_chunk_size)
    local_blocks = 1 + cdiv(q_seqlens - q_tokens_in_first_block, attn_chunk_size)

    # Once we know the number of local blocks we can compute the request spans
    #  for each batch idx, we can figure out the number of "virtual" requests we
    #  have to make,
    # For the above example we would get:
    #   seqlens_q_local = [2, 2, 1, 4, 4, 1, 4, 1]
    #
    # First Get batched arange. (E.g., [2, 4, 2] -> [0, 1, 0, 1, 2, 3, 0, 1])
    #   (TODO: max a utility to share this code with _prepare_inputs)
    # arange step 1. [2, 4, 2] -> [2, 6, 8]
    cu_num_blocks = np.cumsum(local_blocks)
    virtual_batches = cu_num_blocks[-1]
    # arange step 2. [2, 6, 8] -> [0, 0, 2, 2, 2, 2, 6, 6]
    block_offsets = np.repeat(cu_num_blocks - local_blocks, local_blocks)
    # arange step 3. [0, 1, 0, 1, 2, 3, 0, 1]
    arange = np.arange(virtual_batches, dtype=np.int32) - block_offsets
    # also compute reverse arange (i.e. [1, 0, 3, 2, 1, 0, 1, 0])
    rarange = np.repeat(local_blocks, local_blocks) - arange - 1
    # Then we can compute the seqlens_q_local, handling the fact that the
    #  first and last blocks could be partial
    seqlens_q_local = np.repeat(q_seqlens - q_tokens_in_first_block, local_blocks)
    # set the first block since this may be a partial block
    seqlens_q_local[arange == 0] = q_tokens_in_first_block
    # set the remaining blocks
    seqlens_q_local[arange > 0] = np.minimum(
        seqlens_q_local - attn_chunk_size * (arange - 1), attn_chunk_size
    )[arange > 0]

    # convert from q_seqlens to cu_seqlens_q
    cu_seqlens_q_local = np.empty(virtual_batches + 1, dtype=np.int32)
    np.cumsum(seqlens_q_local, out=cu_seqlens_q_local[1:])
    cu_seqlens_q_local[0] = 0

    # compute the seqlens_k_local,
    #  basically a full local attention block for all but the last block in each
    #  batch
    # For our example this will be:
    #   seqlens_k_local = [4, 2, 4, 4, 4, 1, 4, 1]
    seqlens_k_local = np.full(cu_num_blocks[-1], attn_chunk_size, dtype=np.int32)
    seqlens_k_local[cu_num_blocks - 1] = tokens_in_last_block
    num_computed_tokens_local = seqlens_k_local - seqlens_q_local

    k_seqstarts_absolute = np.repeat(seq_lens_np, local_blocks) - (
        rarange * attn_chunk_size + np.repeat(tokens_in_last_block, local_blocks)
    )
    # For the example the local attention blocks start at:
    #                           _b0_  _____b1_____  _b2_
    #   k_seqstarts_absolute = [0, 4, 4, 8, 12, 16, 4, 8]
    block_starts = k_seqstarts_absolute // block_size
    assert attn_chunk_size % block_size == 0, (
        f"attn_chunk_size {attn_chunk_size} is not "
        f"divisible by block_size {block_size}"
    )
    pages_per_local_batch = attn_chunk_size // block_size

    # Create a block_table for the local attention blocks
    # For out example if we have a block-table like (assuming block_size=2):
    #   block_table = [
    #     [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9],  < batch 0
    #     [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],  < batch 1
    #     [20, 21, 22, 23, 24, 25, 26, 27, 28, 29],  < batch 2
    #   ]
    # Then for the local batches we would want a block-table like
    #   block_table_local = [
    #     [  0,  1 ], < local-batch 0, (batch 0, starting from k[0])
    #     [  2,  3 ], < local-batch 1, (batch 0, starting from k[4])
    #     [ 12, 13 ], < local-batch 2, (batch 1, starting from k[4])
    #     [ 14, 15 ], < local-batch 3, (batch 1, starting from k[8])
    #     [ 16, 17 ], < local-batch 4, (batch 1, starting from k[12])
    #     [ 18, 19 ], < local-batch 5, (batch 1, starting from k[16])
    #     [ 22, 23 ], < local-batch 6, (batch 2, starting from k[4])
    #     [ 24, 25 ], < local-batch 7, (batch 2, starting from k[8])
    #   ]
    block_indices = block_starts[:, None] + np.arange(
        pages_per_local_batch, dtype=np.int32
    )
    block_indices = block_indices.reshape(-1).clip(max=block_table.shape[1] - 1)
    batch_indices = np.repeat(
        np.arange(actual_batch_size, dtype=np.int32),
        local_blocks * pages_per_local_batch,
    )
    block_table_local = block_table[batch_indices, block_indices].view(
        virtual_batches, -1
    )

    query_start_loc_cpu = torch.from_numpy(cu_seqlens_q_local)
    seq_lens_cpu = torch.from_numpy(seqlens_k_local)
    max_seq_len = int(seq_lens_cpu.max())

    return CommonAttentionMetadata(
        query_start_loc_cpu=query_start_loc_cpu,
        query_start_loc=query_start_loc_cpu.to(device=device, non_blocking=True),
        seq_lens_cpu=seq_lens_cpu,
        seq_lens=seq_lens_cpu.to(device=device, non_blocking=True),
        num_computed_tokens_cpu=torch.from_numpy(num_computed_tokens_local),
        num_reqs=len(seq_lens_cpu),
        num_actual_tokens=common_attn_metadata.num_actual_tokens,
        max_query_len=seqlens_q_local.max(),
        max_seq_len=max_seq_len,
        block_table_tensor=block_table_local,
        slot_mapping=common_attn_metadata.slot_mapping,
        causal=True,
    )


def subclass_attention_backend(
    name_prefix: str,
    attention_backend_cls: type[AttentionBackend],
    builder_cls: type[AttentionMetadataBuilder[M]],
) -> type[AttentionBackend]:
    """
    Return a new subclass where `get_builder_cls` returns `builder_cls`.
    """
    name: str = name_prefix + attention_backend_cls.__name__  # type: ignore

    return type(
        name, (attention_backend_cls,), {"get_builder_cls": lambda: builder_cls}
    )


def split_decodes_and_prefills(
    common_attn_metadata: CommonAttentionMetadata,
    decode_threshold: int = 1,
) -> tuple[int, int, int, int]:
    """
    Assuming a reordered batch, finds the boundary between prefill and decode
    requests.

    Args:
        common_attn_metadata: CommonAttentionMetadata object containing the
            batch metadata.
        decode_threshold: The maximum query length to be considered a decode.

    Returns:
        num_decodes: The number of decode requests.
        num_prefills: The number of prefill requests.
        num_decode_tokens: The number of tokens in the decode requests.
        num_prefill_tokens: The number of tokens in the prefill requests.
    """
    max_query_len = common_attn_metadata.max_query_len
    num_reqs = common_attn_metadata.num_reqs
    num_tokens = common_attn_metadata.num_actual_tokens
    query_start_loc = common_attn_metadata.query_start_loc_cpu

    if max_query_len <= decode_threshold:
        return num_reqs, 0, num_tokens, 0

    query_lens = query_start_loc[1:] - query_start_loc[:-1]
    is_prefill = query_lens > decode_threshold
    if not torch.any(is_prefill):
        return num_reqs, 0, num_tokens, 0

    first_prefill = is_prefill.int().argmax(dim=-1).item()
    assert torch.all(query_lens[first_prefill:] > decode_threshold)
    assert torch.all(query_lens[:first_prefill] <= decode_threshold)
    num_decodes = first_prefill
    num_prefills = num_reqs - num_decodes
    num_decode_tokens = query_start_loc[first_prefill].item()
    num_prefill_tokens = num_tokens - num_decode_tokens
    return (num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens)


def reorder_batch_to_split_decodes_and_prefills(
    input_batch: "InputBatch",
    scheduler_output: "SchedulerOutput",
    decode_threshold: int = 1,
) -> bool:
    """
    Reorders the batch to split into prefill and decode requests; places all
    requests with <= decode_threshold tokens at the front of the batch.

    Returns:
        True if the batch was modified, False otherwise.
    """
    # We now want to reorder the batch so that the "decode" requests are at
    # the front and the "prefill" requests are at the back using the least
    # amount of swaps possible. (NOTE for now we loosely use "decode" to mean
    # requests where attention is likely memory-bound and "prefill" to mean
    # requests where attention is likely compute-bound, TODO(lucas): figure out
    # a better naming here)
    decodes = []
    prefills = []
    num_decode_tokens = 0
    num_prefill_tokens = 0

    for i, req_id in enumerate(input_batch.req_ids):
        num_tokens = scheduler_output.num_scheduled_tokens[req_id]
        # for now treat 1 scheduled token as "decode" even if its not,
        # we should update this to something like < 8 in the future but
        # currently the TritonMLA._forward_decode only supports
        # num_tokens = 1
        if num_tokens <= decode_threshold:
            decodes.append(i)
            num_decode_tokens += num_tokens
        else:
            prefills.append(i)
            num_prefill_tokens += num_tokens

    # We hope that this is fairly minimal since decodes
    # should be around for a number of iterations so hopefully they are
    # relatively stationary (and new request are generally appended to the
    # persistent batch so already should be at the back)
    # To achieve this we loop over the decodes in descending order and
    # the prefills in ascending order. We swap decodes from the  "back"
    # i.e. past where the last decode should be in the reodorered with
    # prefills from the front of the batch.
    # `decodes` and `prefills` are already in ascending order just based on
    # the above loop
    num_decodes = len(decodes)
    num_prefills = len(prefills)
    modified_batch = False

    for i in range(1, min(num_decodes, num_prefills) + 1):
        # If the decode is at the "back" of the batch, i, we can swap it
        # with the prefill closest to the front of the batch
        decode_idx = decodes[num_decodes - i]
        if decode_idx < num_decodes:
            break

        input_batch.swap_states(prefills[i - 1], decode_idx)
        modified_batch = True

    return modified_batch


KV_SHARING_FAST_PREFILL_METADATA_FIELDS = [
    ("logits_indices_padded", Optional[torch.Tensor], None),
    ("num_logits_indices", int, 0),
]


def subclass_attention_metadata(
    name_prefix: str,
    metadata_cls: Any,
    fields: list[tuple[str, Any, Any]],
) -> Any:
    """
    Return a new subclass of `metadata_cls` with additional fields
    """
    name: str = name_prefix + metadata_cls.__name__  # type: ignore
    Wrapped = make_dataclass(name, fields, bases=(metadata_cls,))
    return Wrapped


def make_kv_sharing_fast_prefill_attention_metadata(
    metadata_cls: Any,
) -> Any:
    """
    Return a new subclass of `metadata_cls` for fast prefill
    """
    return subclass_attention_metadata(
        name_prefix="KVSharingFastPrefill",
        metadata_cls=metadata_cls,
        fields=KV_SHARING_FAST_PREFILL_METADATA_FIELDS,
    )


def estimate_compute_complexity(query_len: int, seq_len: int) -> float:
    """
    Estimate the compute complexity for attention computation.

    Args:
        query_len: Length of query sequence
        seq_len: Length of key/value sequence (context length)

    Returns:
        Estimated relative compute complexity
    """
    # For decode (query_len = 1): O(seq_len) complexity
    # For prefill: O(query_len * seq_len) complexity
    if query_len == 1:
        return float(seq_len)
    else:
        # For prefill, attention complexity is quadratic in query length
        # and linear in context length
        return float(query_len * seq_len)


def analyze_workload(
    query_lens: torch.Tensor, seq_lens: torch.Tensor, decode_threshold: int = 1
) -> UbatchWorkloadInfo:
    """
    Analyze the workload characteristics for intelligent ubatch splitting.

    Args:
        query_lens: Query lengths for each request
        seq_lens: Total sequence lengths for each request
        decode_threshold: Threshold for considering a request as decode

    Returns:
        UbatchWorkloadInfo containing workload analysis
    """
    num_requests = len(query_lens)
    total_tokens = int(torch.sum(query_lens))

    # Classify requests as decode or prefill
    is_decode = query_lens <= decode_threshold
    decode_requests = int(torch.sum(is_decode))
    prefill_requests = num_requests - decode_requests

    decode_tokens = int(torch.sum(query_lens[is_decode]))
    prefill_tokens = total_tokens - decode_tokens

    logger.debug(f"query_lens: {query_lens}, seq_lens: {seq_lens}")

    # Compute complexity estimates
    compute_complexities = torch.tensor(
        [
            estimate_compute_complexity(q_len.item(), s_len.item())
            for q_len, s_len in zip(query_lens, seq_lens)
        ],
        dtype=torch.float32,
    )

    return UbatchWorkloadInfo(
        total_requests=num_requests,
        total_tokens=total_tokens,
        decode_requests=decode_requests,
        prefill_requests=prefill_requests,
        decode_tokens=decode_tokens,
        prefill_tokens=prefill_tokens,
        query_lens=query_lens,
        compute_complexities=compute_complexities,
    )


def create_balanced_ubatch_slices(
    workload_info: UbatchWorkloadInfo,
    num_ubatches: int = 2,
    balance_strategy: str = "compute_complexity",
) -> list[UbatchSlice]:
    """
    Create balanced ubatch slices based on workload characteristics.

    This function supports arbitrary number of ubatches and maintains consecutive
    slices required by the existing UbatchSlice infrastructure. For large batches,
    it uses optimized algorithms to avoid performance bottlenecks.

    Args:
        workload_info: Analysis of the current workload
        num_ubatches: Number of micro-batches to create
        balance_strategy: Strategy for balancing ("compute_complexity" or "tokens")

    Returns:
        List of UbatchSlice objects for balanced micro-batches
    """
    logger.debug(
        f"[UBatch Balance] Creating {num_ubatches} balanced ubatch slices. "
        f"Total requests: {workload_info.total_requests}, "
        f"Total tokens: {workload_info.total_tokens}, "
        f"Decode/Prefill requests: {workload_info.decode_requests}/{workload_info.prefill_requests}, "
        f"Strategy: {balance_strategy}"
    )

    num_requests = workload_info.total_requests

    if num_requests < num_ubatches:
        logger.debug(
            f"[UBatch Balance] Few requests ({num_requests} < {num_ubatches}), creating single request ubatches"
        )
        # If we have fewer requests than ubatches, create one ubatch per request
        return _create_single_request_ubatches(workload_info)

    # Check if this is a mixed workload that needs intelligent splitting
        #has_mixed_workload = (
        #    workload_info.prefill_requests > 0 and workload_info.decode_requests > 0
        #)

        #if not has_mixed_workload:
        #    logger.debug(
        #        f"[UBatch Balance] Uniform workload detected, using simple consecutive splitting"
        #    )
        #    # For uniform workloads (all decode or all prefill), use simple consecutive splitting
        #    return _create_simple_consecutive_ubatch_slices(workload_info, num_ubatches)

        #logger.debug(
        #    f"[UBatch Balance] Mixed workload detected, using balanced consecutive splitting"
        #)
    # For mixed workloads, use balanced splitting
    return _create_balanced_consecutive_ubatch_slices(
        workload_info, num_ubatches, balance_strategy
    )


def _create_single_request_ubatches(
    workload_info: UbatchWorkloadInfo,
) -> list[UbatchSlice]:
    """Create one ubatch per request when we have very few requests."""
    ubatch_slices = []
    token_offset = 0

    for i in range(workload_info.total_requests):
        request_tokens = int(workload_info.query_lens[i])

        ubatch_slice = UbatchSlice(
            request_slice=slice(i, i + 1),
            token_slice=slice(token_offset, token_offset + request_tokens),
            compute_complexity=float(workload_info.compute_complexities[i]),
            query_lens=workload_info.query_lens[i : i + 1],
            has_prefill=bool(workload_info.query_lens[i] > 1),
            max_query_len=int(workload_info.query_lens[i]),
        )
        ubatch_slices.append(ubatch_slice)
        token_offset += request_tokens

    return ubatch_slices


def _create_simple_consecutive_ubatch_slices(
    workload_info: UbatchWorkloadInfo, num_ubatches: int
) -> list[UbatchSlice]:
    """Create consecutive ubatch slices for uniform workloads."""
    num_requests = workload_info.total_requests
    logger.debug(
        f"[UBatch Simple] Creating {num_ubatches} simple consecutive slices for {num_requests} requests"
    )

    ubatch_slices = []

    for i in range(num_ubatches):
        # Calculate request boundaries
        request_start = (i * num_requests) // num_ubatches
        request_end = ((i + 1) * num_requests) // num_ubatches

        # Calculate token boundaries based on actual tokens in this request range
        if i == 0:
            token_start = 0
        else:
            token_start = int(torch.sum(workload_info.query_lens[:request_start]))
        token_end = int(torch.sum(workload_info.query_lens[:request_end]))

        # Calculate properties for this slice
        slice_query_lens = workload_info.query_lens[request_start:request_end]
        slice_complexities = workload_info.compute_complexities[
            request_start:request_end
        ]

        ubatch_slice = UbatchSlice(
            request_slice=slice(request_start, request_end),
            token_slice=slice(token_start, token_end),
            compute_complexity=float(torch.sum(slice_complexities)),
            query_lens=slice_query_lens,
            has_prefill=bool(torch.any(slice_query_lens > 1)),
            max_query_len=int(torch.max(slice_query_lens)),
        )
        ubatch_slices.append(ubatch_slice)
        logger.debug(
            f"[UBatch Simple] Ubatch {i}: requests [{request_start}:{request_end}], "
            f"tokens [{token_start}:{token_end}], complexity: {ubatch_slice.compute_complexity:.2f}, "
            f"has_prefill: {ubatch_slice.has_prefill}"
        )

    return ubatch_slices


def _create_balanced_consecutive_ubatch_slices(
    workload_info: UbatchWorkloadInfo,
    num_ubatches: int,
    balance_strategy: str = "compute_complexity",
) -> list[UbatchSlice]:
    """
    Create balanced ubatch slices with consecutive request indices using dynamic programming.

    This ensures that each ubatch contains consecutive request indices (e.g., 1,2,3 not 1,2,4)
    while attempting to balance the load across ubatches based on the selected strategy.
    """
    num_requests = workload_info.total_requests

    # Handle edge cases
    #if num_requests <= num_ubatches:
    #    return _create_single_request_ubatches(workload_info)

    ## Use simple approach for small batches
    #if num_requests <= 32:
    #    return _create_simple_consecutive_ubatch_slices(workload_info, num_ubatches)

    # Choose weights based on strategy
    if balance_strategy == "compute_complexity":
        weights = workload_info.compute_complexities
    else:  # "tokens"
        weights = workload_info.query_lens.float()

    # Use a greedy approach to find balanced consecutive splits
    # Calculate prefix sums for efficient range sum queries
    prefix_weights = torch.cumsum(weights, dim=0)
    total_weight = float(prefix_weights[-1])
    target_weight_per_ubatch = total_weight / num_ubatches

    # Find split points that create roughly balanced consecutive segments
    split_points = [0]  # Start with first request
    current_target = target_weight_per_ubatch

    for i in range(1, num_requests):
        current_weight = float(prefix_weights[i - 1])  # Weight up to current position

        # If we've reached our target weight and we're not at the last ubatch
        if current_weight >= current_target and len(split_points) < num_ubatches:
            split_points.append(i)
            current_target += target_weight_per_ubatch

    # Ensure we end at the last request
    if split_points[-1] != num_requests:
        split_points.append(num_requests)

    # Create ubatch slices from consecutive segments
    ubatch_slices = []

    for i in range(len(split_points) - 1):
        start_idx = split_points[i]
        end_idx = split_points[i + 1]

        # Calculate token boundaries
        if start_idx == 0:
            token_start = 0
        else:
            token_start = int(torch.sum(workload_info.query_lens[:start_idx]))
        token_end = int(torch.sum(workload_info.query_lens[:end_idx]))

        # Get slice properties
        slice_query_lens = workload_info.query_lens[start_idx:end_idx]
        slice_complexities = workload_info.compute_complexities[start_idx:end_idx]

        ubatch_slice = UbatchSlice(
            request_slice=slice(start_idx, end_idx),
            token_slice=slice(token_start, token_end),
            compute_complexity=float(torch.sum(slice_complexities)),
            query_lens=slice_query_lens,
            has_prefill=bool(torch.any(slice_query_lens > 1)),
            max_query_len=int(torch.max(slice_query_lens)),
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
