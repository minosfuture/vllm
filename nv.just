# -------
# CONFIG
# ------

MODEL := "nvidia/DeepSeek-R1-0528-FP4-v2"
PREC := "fp4"
LYRIS_IFNAME := "enP22p3s0f1np1"
PTYCHE_IFNAME := "enP6p3s0f1np1"

# ---
# ENV
# ---

# NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME=enP22p3s0f1np1 \
# GLOO_SOCKET_IFNAME=enP22p3s0f1np1 \
# NCCL_SOCKET_IFNAME=enP22p3s0f1np1 \
# UCX_IB_ROCE_REACHABILITY_MODE=local_subnet \
SYSTEM_ENV := '''NVIDIA_GDRCOPY=1 \
  NVSHMEM_IB_ENABLE_IBGDA=1 \
  VLLM_SKIP_P2P_CHECK=1 \
  NCCL_CUMEM_ENABLE=1 \
  NCCL_MNNVL_ENABLE=1 \
  NCCL_NVLS_ENABLE=1 '''


COMMON_VLLM_ENV := '''\
  VLLM_RANDOMIZE_DP_DUMMY_INPUTS=1 \
  VLLM_ATTENTION_BACKEND=FLASHINFER_MLA \
  VLLM_USE_FLASHINFER_MOE_FP4=1 \
  VLLM_USE_TRTLLM_RAGGED_DEEPSEEK_PREFILL=1 \
  VLLM_FLASHINFER_MOE_BACKEND=latency \
  VLLM_USE_NCCL_SYMM_MEM=1 '''

# Currently there is no prefill or decode specific environment variables
PREFILL_VLLM_ENV := SYSTEM_ENV + COMMON_VLLM_ENV 
PREFILL_PD_VLLM_ENV := PREFILL_VLLM_ENV + PD_VLLM_ENV

DECODE_VLLM_ENV := SYSTEM_ENV + COMMON_VLLM_ENV
DECODE_PD_VLLM_ENV := DECODE_VLLM_ENV + PD_VLLM_ENV

# ---------
# vLLM Args
# ---------

COMMON_VLLM_ARGS := '''
  --kv-cache-dtype fp8 \
    --tensor-parallel-size 1 \
    --pipeline-parallel-size 1 \
    --enable-expert-parallel \
    --data-parallel-rpc-port 13345 \
    --max-model-len 4096 \
    --disable-uvicorn-access-log \
    --no-enable-prefix-caching \
    --port 8000 \
    --async-scheduling '''

PREFILL_VLLM_ARGS := COMMON_VLLM_ARGS + '''\
  --async-scheduling \
  --disable_custom_all_reduce \
  --disable-uvicorn-access-log \
  --disable_nccl_for_dp_synchronization \
  --enable-expert-parallel \
  --kv-cache-dtype fp8 \
  --tensor-parallel-size 1 \
  --trust-remote-code \
  --all2all-backend allgather_reducescatter \
  --compilation_config.custom_ops+=+quant_fp8,+rms_norm,+rotary_embedding \
  --compilation_config.pass_config.enable_attn_fusion true \
  --compilation_config.pass_config.enable_fi_allreduce_fusion true \
  --compilation_config.pass_config.enable_noop true \
  --gpu-memory-utilization 0.85 \
  --max-model-len 4096 \
  --max-num-batched-tokens 65536 \
  --max-num-seqs 1024 \
  --no-enable-prefix-caching \
  --swap-space 16 \
  --trust-remote-code \
  --data-parallel-size 2 \
  --enforce-eager \
  --offload-group-size 2 \
  --offload-num-in-group 1 \
  --offload-prefetch-step 1 \
'''

PREFILL_PD_VLLM_ARGS := PREFILL_VLLM_ARGS + PD_VLLM_ARGS

DECODE_VLLM_ARGS := COMMON_VLLM_ARGS + '''\
  --all2all-backend flashinfer_all2allv \
  --compilation-config.cudagraph_mode FULL_DECODE_ONLY \
  --compilation_config.custom_ops+=+rms_norm,+rotary_embedding \
  --data-parallel-hybrid-lb \
  --gpu-memory-utilization 0.9 \
  --data-parallel-size-local 4 \
  --stream-interval 50 \
  --max-num-seqs 1024 \
  --max-num-batched-tokens 8192 \
  --max-cudagraph-capture-size 1024 '''

DECODE_PD_VLLM_ARGS := DECODE_VLLM_ARGS + PD_VLLM_ARGS

# ---
# PD
# ---

PD_VLLM_ENV := '''\
  VLLM_NIXL_SIDE_CHANNEL_HOST=`hostname -i` \
  VLLM_NIXL_SIDE_CHANNEL_PORT=5600 \
  VLLM_NIXL_ABORT_REQUEST_TIMEOUT=300 '''

PD_VLLM_ARGS := '''\
  --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both", "kv_load_failure_policy":"fail"}' '''

# ----
# NSYS
# ----

NSYS_COMMAND := '''\
  VLLM_TORCH_CUDA_PROFILE=1 \
  VLLM_TORCH_PROFILER_USE_GZIP=0 \
  VLLM_TORCH_PROFILER_DUMP_CUDA_TIME_TOTAL=0 \
  VLLM_NVTX_SCOPES_FOR_PROFILING=1 \
  nsys profile \
  --trace=nvtx,cuda \
  --force-overwrite true \
  --gpu-metrics-devices=all \
  --trace-fork-before-exec=true \
  --cuda-graph-trace=node \
  --capture-range=cudaProfilerApi \
  --capture-range-end "repeat" \
  '''


# -------
# Recipes
# -------

prefill-master PMA DPS:
  {{PREFILL_VLLM_ENV}} {{NSYS_COMMAND}} -o ds-{{PREC}}-prefill-master.nsys-rep vllm serve {{MODEL}} \
    {{PREFILL_VLLM_ARGS}} \
    --data-parallel-address {{PMA}} \
    --data-parallel-size {{DPS}} \
    2>&1 | tee prefill-master.log

prefill-worker PMA DPSR DPS:
  {{PREFILL_VLLM_ENV}} {{NSYS_COMMAND}} -o ds-{{PREC}}-prefill-worker-{{DPSR}}.nsys-rep vllm serve {{MODEL}} \
    {{PREFILL_VLLM_ARGS}} \
    --data-parallel-address {{PMA}} \
    --data-parallel-start-rank {{DPSR}} \
    --data-parallel-size {{DPS}} \
    --headless \
    2>&1 | tee prefill-worker-{{DPSR}}.log

decode-master DMA DPS:
  {{DECODE_VLLM_ENV}} {{NSYS_COMMAND}} -o ds-{{PREC}}-decode-master.nsys-rep vllm serve {{MODEL}} \
    {{DECODE_VLLM_ARGS}} \
    --data-parallel-address {{DMA}} \
    --data-parallel-size {{DPS}} \
    2>&1 | tee decode-master.log

decode-worker DMA DPSR DPS:
  {{DECODE_VLLM_ENV}} {{NSYS_COMMAND}} -o ds-{{PREC}}-decode-worker-{{DPSR}}.nsys-rep vllm serve {{MODEL}} \
    {{DECODE_VLLM_ARGS}} \
    --data-parallel-address {{DMA}} \
    --data-parallel-start-rank {{DPSR}} \
    --data-parallel-size {{DPS}} \
    2>&1 | tee decode-worker-{{DPSR}}.log

prefill-master-pd PMA DPS:
  {{PREFILL_PD_VLLM_ENV}} {{NSYS_COMMAND}} -o ds-{{PREC}}-prefill-master.nsys-rep vllm serve {{MODEL}} \
    {{PREFILL_PD_VLLM_ARGS}} \
    --data-parallel-address {{PMA}} \
    --data-parallel-size {{DPS}} \
    2>&1 | tee prefill-master-pd.log

prefill-worker-pd PMA DPSR DPS:
  {{PREFILL_PD_VLLM_ENV}} {{NSYS_COMMAND}} -o ds-{{PREC}}-prefill-worker-{{DPSR}}.nsys-rep vllm serve {{MODEL}} \
    {{PREFILL_PD_VLLM_ARGS}} \
    --data-parallel-address {{PMA}} \
    --data-parallel-start-rank {{DPSR}} \
    --data-parallel-size {{DPS}} \
    2>&1 | tee prefill-worker-{{DPSR}}.log

decode-master-pd DMA DPS:
  {{DECODE_PD_VLLM_ENV}} \
    VLLM_TORCH_PROFILER_DIR=./profile/ \
    vllm serve {{MODEL}} \
    {{DECODE_PD_VLLM_ARGS}} \
    --data-parallel-address {{DMA}} \
    --data-parallel-size {{DPS}} \
    2>&1 | tee decode-master-pd.log

decode-worker-pd DMA DPSR DPS:
  {{DECODE_PD_VLLM_ENV}} \
    VLLM_TORCH_PROFILER_DIR=./profile/ \
    vllm serve {{MODEL}} \
    {{DECODE_PD_VLLM_ARGS}} \
    --data-parallel-address {{DMA}} \
    --data-parallel-start-rank {{DPSR}} \
    --data-parallel-size {{DPS}} \
    2>&1 | tee decode-worker-pd-{{DPSR}}.log

proxy PMA DMA:
  python3 /scratch/toy_proxy_server.py \
    --port 8192 \
    --prefiller-hosts {{PMA}} \
    --prefiller-ports 8087 \
    --decoder-hosts {{DMA}} \
    --decoder-ports 8087 \
    2>&1 | tee proxy.log

router PA1 PA2 PA3 PA4 DA1 DA2:
  #!/usr/bin/env bash
  pushd /vllm-workspace/router
  RUST_LOG=warn \
    cargo run --release -- \
    --policy round_robin \
    --vllm-pd-disaggregation \
    --max-concurrent-requests 16384 \
    --prefill {{PA1}} \
    --prefill {{PA2}} \
    --prefill {{PA3}} \
    --prefill {{PA4}} \
    --decode {{DA1}} \
    --decode {{DA2}} \
    --host 0.0.0.0 \
    --port 8192 \
    --intra-node-data-parallel-size 4
  popd

bench PMA="" PORT="8087" MC="8096" ISL="1024" OSL="1024" COMMON_PREFIX="0" SKIP_REQS="0":
  #!/usr/bin/env bash
  # Source utils.sh (assumes running from runs/fp4/ directory)
  source ../utils.sh
  if [ "$nsys" = "true" ]; then
  start_profiler_control "{{PMA}}" "{{PORT}}"
  fi
  vllm bench serve \
    --model {{MODEL}} \
    --host {{PMA}} \
    --port {{PORT}} \
    --dataset-name random \
    --ignore-eos \
    --request-rate 100 \
    --num-prompts {{MC}} \
    --max-concurrency {{MC}} \
    --random-input-len {{ISL}} \
    --random-output-len {{OSL}} \
    --ready-check-timeout-sec 0 \
    --common-prefix-len {{COMMON_PREFIX}} \
    --num-skip-requests {{SKIP_REQS}} \
    2>&1 | tee bench.log

accuracy MA="" PORT="8087":
  lm_eval --model local-completions --tasks gsm8k --trust_remote_code --batch_size auto \
    --model_args base_url=http://{{MA}}:{{PORT}}/v1/completions,pretrained={{MODEL}},model={{MODEL}},add_bos_token=True,tokenized_requests=False,tokenizer_backend=None,num_concurrent=1024,timeout=60000,max_retries=5 \
    2>&1 | tee eval.log

kill:
  pkill -9 -f VLLM::EngineCore || echo "All processes stopped"
