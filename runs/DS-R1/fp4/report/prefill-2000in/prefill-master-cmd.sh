NVIDIA_GDRCOPY=1 \
  NVSHMEM_IB_ENABLE_IBGDA=1 \
  VLLM_SKIP_P2P_CHECK=1 \
  NCCL_CUMEM_ENABLE=1 \
  NCCL_MNNVL_ENABLE=1 \
  NCCL_NVLS_ENABLE=1 \
  VLLM_RANDOMIZE_DP_DUMMY_INPUTS=1 \
  VLLM_USE_FLASHINFER_MOE_FP4=1 \
  VLLM_FLASHINFER_MOE_BACKEND=latency \
  VLLM_ENABLE_FUSED_MOE_ACTIVATION_CHUNKING=0 \
  VLLM_USE_NCCL_SYMM_MEM=1 \
  VLLM_ENABLE_MOE_DP_CHUNK=0 \
  VLLM_NIXL_SIDE_CHANNEL_HOST=$(hostname -i) \
  VLLM_NIXL_SIDE_CHANNEL_PORT=5700 \
  VLLM_NIXL_ABORT_REQUEST_TIMEOUT=600 numactl --cpunodebind=1 --membind=1 vllm serve /ds-models/DeepSeek-R1-0528-NVFP4-v2 --profiler-config.torch_profiler_dir=./profile/ \
  --attention-config.backend FLASHINFER_MLA \
  --attention-config.use_trtllm_ragged_deepseek_prefill=true \
  --kv-cache-dtype fp8 \
  --enable-expert-parallel \
  --data-parallel-rpc-port 13345 \
  --max-model-len 2148 \
  --disable-uvicorn-access-log \
  --port 8088 \
  --trust-remote-code \
  --async-scheduling \
  --disable_custom_all_reduce \
  --disable_nccl_for_dp_synchronization \
  --no-enable-prefix-caching \
  --all2all-backend allgather_reducescatter \
  --gpu-memory-utilization 0.85 \
  --max-num-batched-tokens 65536 \
  --max-num-seqs 1024 \
  --swap-space 16 \
  --data-parallel-size 2 \
  --enforce-eager \
  --offload-group-size 2 \
  --offload-num-in-group 1 \
  --offload-prefetch-step 1 \
  \
  --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both", "kv_load_failure_policy":"fail"}' --data-parallel-address 10.244.11.49 --data-parallel-size 2 2>&1 | tee prefill-master-pd.log
