NVIDIA_GDRCOPY=1 \
  NVSHMEM_IB_ENABLE_IBGDA=1 \
  VLLM_SKIP_P2P_CHECK=0 \
  NCCL_CUMEM_ENABLE=1 \
  NCCL_MNNVL_ENABLE=2 \
  NCCL_DEBUG=INFO \
  NCCL_DEBUG_FILE=./nccl-debug-%h-%p.log \
  NCCL_NVLS_ENABLE=1 \
  VLLM_RANDOMIZE_DP_DUMMY_INPUTS=1 \
  VLLM_ATTENTION_BACKEND=FLASHINFER_MLA \
  VLLM_USE_FLASHINFER_MOE_FP4=1 \
  VLLM_USE_TRTLLM_RAGGED_DEEPSEEK_PREFILL=1 \
  VLLM_FLASHINFER_MOE_BACKEND=latency \
  VLLM_ENABLE_FUSED_MOE_ACTIVATION_CHUNKING=0 \
  VLLM_USE_NCCL_SYMM_MEM=1 \
  VLLM_LOG_STATS_INTERVAL=1 \
  VLLM_MLA_FP8_PROJ=1 \
  VLLM_DEEPEP_LOW_LATENCY_ALLOW_NVLINK=1 \
  VLLM_DEEPEP_LOW_LATENCY_USE_MNNVL=1 \
  VLLM_DEEPEP_BUFFER_SIZE_MB=0 \
  VLLM_EP_USE_SBO=0 \
  VLLM_MOE_DP_CHUNK_SIZE=1024 \
  VLLM_DEEPEPLL_NVFP4_DISPATCH=1 \
  VLLM_ENABLE_FUSED_MOE_ACTIVATION_CHUNKING=0 \
  VLLM_V1_OUTPUT_PROC_CHUNK_SIZE=2048 \
  VLLM_NIXL_SIDE_CHANNEL_HOST=$(hostname -i) \
  VLLM_NIXL_SIDE_CHANNEL_PORT=5700 \
  VLLM_NIXL_ABORT_REQUEST_TIMEOUT=600 vllm serve /ds-models/DeepSeek-R1-0528-NVFP4-v2 --profiler-config.torch_profiler_dir=./profile/ \
  --kv-cache-dtype fp8 \
  --enable-expert-parallel \
  --data-parallel-rpc-port 13345 \
  --max-model-len 4096 \
  --disable-uvicorn-access-log \
  --port 8088 \
  --trust-remote-code \
  --async-scheduling \
  --all2all-backend allgather_reducescatter \
  --compilation-config.cudagraph_mode FULL_DECODE_ONLY \
  --compilation_config.custom_ops+=+rms_norm,+rotary_embedding \
  --data-parallel-hybrid-lb \
  --kv_cache_memory_bytes 119519622533 \
  --data-parallel-size-local 4 \
  --stream-interval 50 \
  --max-num-seqs 1024 \
  --max-num-batched-tokens 1024 \
  --cudagraph-capture-sizes 128 512 1024 \
  --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both", "kv_load_failure_policy":"fail"}' --data-parallel-address 10.244.2.151 --data-parallel-start-rank 4 --data-parallel-size 8 2>&1 | tee decode-worker-pd-4.log
