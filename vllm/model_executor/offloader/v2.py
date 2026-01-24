# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from
# https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/utils/offloader.py
"""OffloaderV2: CPU offloading with async prefetching.

This version uses static buffers and stream synchronization (instead of
CUDA events) for torch.compile + CUDA graph compatibility.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable, Generator
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.func import functional_call

# Import v2_ops to register custom ops at module load time
import vllm.model_executor.offloader.v2_ops  # noqa: F401
from vllm.logger import init_logger
from vllm.model_executor.offloader.base import BaseOffloader
from vllm.utils.platform_utils import is_pin_memory_available

logger = init_logger(__name__)

_SubmoduleAccessor = Callable[[nn.Module], nn.Module]
_WhitelistParamNamesCreator = Callable[[nn.Module], list[str]]


@dataclass
class ParamInfo:
    """Metadata about an offloaded parameter."""

    name: str
    shape: tuple[int, ...]
    dtype: torch.dtype

    @property
    def key(self) -> tuple[tuple[int, ...], torch.dtype]:
        """Unique key for buffer pool grouping."""
        return (self.shape, self.dtype)

    @property
    def num_bytes(self) -> int:
        """Size in bytes."""
        numel = 1
        for dim in self.shape:
            numel *= dim
        return numel * torch.tensor([], dtype=self.dtype).element_size()


class StaticBufferPool:
    """Pre-allocated GPU buffer pool for offloaded parameters.

    Allocates slot_capacity copies of each unique parameter shape,
    allowing for double/triple buffering during prefetch.

    Buffer slots are reused circularly: layer N uses slot (N % slot_capacity).
    """

    def __init__(
        self,
        param_infos: list[ParamInfo],
        slot_capacity: int,
        device: torch.device,
    ):
        self.slot_capacity = slot_capacity
        self.total_bytes = 0
        self._device = device

        # Group by (shape, dtype) - only allocate unique shapes
        unique_params: dict[tuple, ParamInfo] = {}
        for info in param_infos:
            if info.key not in unique_params:
                unique_params[info.key] = info

        # Allocate buffers: key -> list of tensors (one per slot)
        self._buffers: dict[tuple, list[torch.Tensor]] = {}
        for key, info in unique_params.items():
            slot_tensors = []
            for _ in range(slot_capacity):
                buf = torch.empty(
                    info.shape,
                    dtype=info.dtype,
                    device=device,
                )
                slot_tensors.append(buf)
                self.total_bytes += info.num_bytes
            self._buffers[key] = slot_tensors

        logger.debug(
            "[StaticBufferPool] Allocated %d unique shapes, "
            "%d slots each, total %.4f GB",
            len(unique_params),
            slot_capacity,
            self.total_bytes / 1e9,
        )

    def get_buffer(
        self,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        slot_idx: int,
    ) -> torch.Tensor:
        """Get a static buffer for the given shape/dtype/slot."""
        key = (shape, dtype)
        return self._buffers[key][slot_idx % self.slot_capacity]


class OffloaderV2(BaseOffloader):
    """Advanced offloader with group-based selection and async prefetching.

    Uses static buffers and stream synchronization for torch.compile and
    CUDA graph compatibility.

    Args:
        group_size: Group every N layers together.
        num_in_group: Offload this many layers per group (last N of each group).
        prefetch_step: Number of layers to prefetch ahead.
        mode: Offload mode ("cpu" is currently supported).
    """

    def __init__(
        self,
        group_size: int,
        num_in_group: int,
        prefetch_step: int,
        mode: str = "cpu",
    ):
        self.group_size = group_size
        self.num_in_group = num_in_group
        self.prefetch_step = prefetch_step
        self.mode = mode

        # Copy stream for async H2D transfers
        self.copy_stream = torch.cuda.Stream()

        # Sync tensor for custom op data dependencies (prevents reordering)
        self._sync_tensor: torch.Tensor | None = None

        # Module offloaders and buffer pool (populated in wrap_modules/post_init)
        self.module_offloaders: list[_ModuleOffloader] = []
        self.buffer_pool: StaticBufferPool | None = None
        self.total_offloaded_bytes = 0

        # Register this instance for custom ops
        from vllm.model_executor.offloader.v2_ops import set_offloader_instance

        set_offloader_instance(self)

    def wrap_modules(
        self,
        modules_generator: Generator[nn.Module, None, None],
        submodule_accessor: _SubmoduleAccessor | None = None,
        whitelist_param_names_creator: _WhitelistParamNamesCreator | None = None,
    ) -> list[nn.Module]:
        """Wrap modules with V2 offloading and prefetching logic."""
        assert len(self.module_offloaders) == 0, (
            "wrap_modules should only be called once"
        )

        all_modules = []
        offload_submodules = []

        for module_index, module in enumerate(modules_generator):
            all_modules.append(module)

            # Select layers to offload based on group pattern
            # Offload last num_in_group layers of each group_size
            if module_index % self.group_size >= self.group_size - self.num_in_group:
                submodule = submodule_accessor(module) if submodule_accessor else module
                whitelist_param_names = (
                    whitelist_param_names_creator(submodule)
                    if whitelist_param_names_creator
                    else [name for name, _ in submodule.named_parameters()]
                )

                offload_submodules.append(submodule)
                self.module_offloaders.append(
                    _ModuleOffloader(
                        mode=self.mode,
                        module=submodule,
                        copy_stream=self.copy_stream,
                        whitelist_param_names=whitelist_param_names,
                        layer_idx=len(self.module_offloaders),
                    )
                )

        for index, submodule in enumerate(offload_submodules):
            self._hook_module_forward(index, submodule)

        return all_modules

    def _hook_module_forward(self, index: int, module: nn.Module):
        """Hook module's forward with torch.compile-compatible sync."""
        original_forward = module.forward
        offloader = self.module_offloaders[index]

        def forward(*args, **kwargs):
            # Temporarily restore original forward to avoid recursion
            module.forward = original_forward

            # Wait for this layer's prefetch to complete
            # Custom op prevents reordering across this point
            torch.ops.vllm.wait_prefetch(self._sync_tensor, index)

            # Get static buffers for this layer
            device_tensors = offloader.get_static_buffers()

            # Execute with the static buffer parameters
            output = functional_call(module, device_tensors, args=args, kwargs=kwargs)

            # Start prefetch for next layer (circular)
            # Pass output to create ordering dependency - compiler cannot reorder
            # this before functional_call since start_prefetch "mutates" output
            next_index = (index + self.prefetch_step) % len(self.module_offloaders)
            # Handle tuple output from functional_call (e.g., (hidden_states, residual))
            output_tensor = output[0] if isinstance(output, tuple) else output
            torch.ops.vllm.start_prefetch(self._sync_tensor, next_index, output_tensor)

            # No explicit offload needed - static buffers are reused implicitly

            # Restore hooked forward
            module.forward = forward
            return output

        module.forward = forward

    def _wait_for_layer(self, layer_idx: int):
        """Called by custom op - wait for copy stream to complete."""
        # wait_stream creates a CUDA graph dependency edge when captured
        torch.cuda.current_stream().wait_stream(self.copy_stream)

    def _start_prefetch(self, layer_idx: int):
        """Called by custom op - start async copy to static buffer."""
        offloader = self.module_offloaders[layer_idx]
        offloader.start_onload_to_static()

    def post_init(self):
        """Allocate static buffer pool and start initial prefetches."""
        # Collect parameter info and finalize offloaders
        param_infos: list[ParamInfo] = []
        device: torch.device | None = None

        for offloader in self.module_offloaders:
            offloader.post_init()
            self.total_offloaded_bytes += offloader.offloaded_bytes
            param_infos.extend(offloader.get_param_infos())
            if device is None:
                device = offloader.device

        if device is None:
            # No modules to offload
            return

        # Create sync tensor on the device
        self._sync_tensor = torch.empty(0, device=device)

        # Allocate static buffer pool
        self.buffer_pool = StaticBufferPool(
            param_infos=param_infos,
            slot_capacity=self.prefetch_step,
            device=device,
        )

        # Assign buffer slots to offloaders (circular assignment)
        for idx, offloader in enumerate(self.module_offloaders):
            slot_idx = idx % self.prefetch_step
            offloader.assign_buffer_slot(self.buffer_pool, slot_idx)

        logger.info_once(
            f"[OffloaderV2] Initialized {len(self.module_offloaders)} modules. "
            f"Total GPU memory saved: {self.total_offloaded_bytes / 1e9:.4f} GB, "
            f"Static buffer pool: {self.buffer_pool.total_bytes / 1e9:.4f} GB "
            f"(group_size={self.group_size}, num_in_group={self.num_in_group}, "
            f"prefetch_step={self.prefetch_step}, mode={self.mode})"
        )

        # Start initial prefetches
        for i in range(min(self.prefetch_step, len(self.module_offloaders))):
            self.module_offloaders[i].start_onload_to_static()


class _ModuleOffloader:
    """Manages offloading for a single module.

    Uses static buffers from a shared pool instead of dynamic allocation.
    """

    def __init__(
        self,
        mode: str,
        module: nn.Module,
        copy_stream: torch.cuda.Stream,
        whitelist_param_names: list[str],
        layer_idx: int,
    ):
        self.mode = mode
        self.module = module
        self.device = next(module.parameters()).device
        self.copy_stream = copy_stream
        self.layer_idx = layer_idx
        self.offloaded_bytes = 0

        assert self.device != torch.device("cpu"), (
            "Module parameters should not already be on CPU "
            "(offloader handles CPU placement)"
        )

        # Buffer pool and slot (assigned in assign_buffer_slot)
        self._buffer_pool: StaticBufferPool | None = None
        self._buffer_slot_idx: int = 0

        param_dict = dict(self.module.named_parameters())
        assert all(name in param_dict for name in whitelist_param_names), (
            f"Whitelist params {whitelist_param_names} not found in module params "
            f"{list(param_dict.keys())}"
        )

        self._param_offloaders = {
            name: _BaseParamOffloader.create(mode, module=module, param_name=name)
            for name in whitelist_param_names
        }

    def post_init(self):
        """Collect total offloaded bytes (offloading already done in __init__)."""
        for param_offloader in self._param_offloaders.values():
            param_offloader.post_init()
            self.offloaded_bytes += param_offloader.offloaded_bytes

    def get_param_infos(self) -> list[ParamInfo]:
        """Get parameter metadata for buffer pool allocation."""
        infos = []
        for name, offloader in self._param_offloaders.items():
            param = offloader._param
            infos.append(
                ParamInfo(
                    name=name,
                    shape=tuple(param.shape),
                    dtype=param.dtype,
                )
            )
        return infos

    def assign_buffer_slot(self, pool: StaticBufferPool, slot_idx: int):
        """Assign this module to a buffer slot in the pool."""
        self._buffer_pool = pool
        self._buffer_slot_idx = slot_idx

    def start_onload_to_static(self):
        """Start async copy from CPU to static GPU buffer."""
        assert self._buffer_pool is not None, "Buffer pool not assigned"

        with torch.cuda.stream(self.copy_stream):
            for name, offloader in self._param_offloaders.items():
                param = offloader._param  # CPU tensor
                buffer = self._buffer_pool.get_buffer(
                    shape=tuple(param.shape),
                    dtype=param.dtype,
                    slot_idx=self._buffer_slot_idx,
                )
                # Async copy from pinned CPU to GPU buffer
                buffer.copy_(param, non_blocking=True)

    def get_static_buffers(self) -> dict[str, torch.Tensor]:
        """Get static GPU buffers for this layer (after sync)."""
        assert self._buffer_pool is not None, "Buffer pool not assigned"

        result = {}
        for name, offloader in self._param_offloaders.items():
            param = offloader._param
            buffer = self._buffer_pool.get_buffer(
                shape=tuple(param.shape),
                dtype=param.dtype,
                slot_idx=self._buffer_slot_idx,
            )
            result[name] = buffer
        return result


class _BaseParamOffloader(ABC):
    """Base class for parameter offloading strategies."""

    @staticmethod
    def create(mode: str, **kwargs) -> "_BaseParamOffloader":
        """Factory method to create appropriate offloader for mode."""
        if mode == "cpu":
            return _CpuParamOffloader(**kwargs)
        else:
            raise ValueError(f"Unknown offload mode: {mode}")

    def __init__(self, module: nn.Module, param_name: str):
        self._module = module
        self._param_name = param_name
        self.offloaded_bytes = 0

    @property
    def _param(self) -> nn.Parameter:
        """Get the parameter being offloaded."""
        return getattr(self._module, self._param_name)

    def post_init(self):
        """Initialize offloading (move parameter to storage)."""
        return

    @abstractmethod
    def create_device_tensor(self) -> torch.Tensor:
        """Create device tensor from offloaded storage."""
        pass


class _CpuParamOffloader(_BaseParamOffloader):
    """Offload parameter to pinned CPU memory."""

    def __init__(self, module: nn.Module, param_name: str):
        super().__init__(module, param_name)
        self._move_param_to_cpu()

    def _move_param_to_cpu(self):
        """Move parameter data to pinned CPU memory (modify param.data in-place)."""
        param = self._param
        pin_memory = is_pin_memory_available()

        self.offloaded_bytes = param.data.numel() * param.data.element_size()

        cpu_data = torch.empty_strided(
            size=param.data.size(),
            stride=param.data.stride(),
            dtype=param.data.dtype,
            layout=param.data.layout,
            device="cpu",
            pin_memory=pin_memory,
        )
        cpu_data.copy_(param.data)

        logger.debug_once(
            f"[OffloaderV2] Offloaded parameter '{self._param_name}': "
            f"shape={tuple(param.shape)}, dtype={param.dtype}, "
            f"size={self.offloaded_bytes / 1e9:.6f} GB, pinned={pin_memory}"
        )

        param.data = cpu_data

    def post_init(self):
        """No-op: offloading already done in __init__."""
        pass

    def create_device_tensor(self) -> torch.Tensor:
        """Load from CPU to GPU (async if pinned).

        Returns a CUDA copy of the parameter (which has CPU data).
        Note: This method is kept for backwards compatibility but is not
        used in the static buffer approach.
        """
        return self._param.to("cuda", non_blocking=True)
