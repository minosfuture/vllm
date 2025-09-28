#!/bin/bash

# vLLM Benchmark Runner Script
# Runs vLLM server with different configurations and executes benchmarks

set -e

# Default configuration
DEFAULT_NUM_PROMPTS=4096
DEFAULT_INPUT_LEN=2048
DEFAULT_OUTPUT_LEN=1024
DEFAULT_MAX_CONCURRENCY=4096
DEFAULT_RESULT_DIR="/home/yming/local/vllm_profile"
DEFAULT_PORT=8000

# Parse command line arguments
NUM_PROMPTS=${1:-$DEFAULT_NUM_PROMPTS}
INPUT_LEN=${2:-$DEFAULT_INPUT_LEN}
OUTPUT_LEN=${3:-$DEFAULT_OUTPUT_LEN}
MAX_CONCURRENCY=${4:-$DEFAULT_MAX_CONCURRENCY}
RESULT_DIR=${5:-$DEFAULT_RESULT_DIR}
PORT=${6:-$DEFAULT_PORT}

# Create timestamp for this run
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_RESULT_DIR="${RESULT_DIR}/mnbk16384_mns1349_results_${TIMESTAMP}"

# Create result directory
mkdir -p "$RUN_RESULT_DIR"

echo "=== vLLM Benchmark Runner ==="
echo "Configuration:"
echo "  - Number of prompts: $NUM_PROMPTS"
echo "  - Input length: $INPUT_LEN"
echo "  - Output length: $OUTPUT_LEN"
echo "  - Max concurrency: $MAX_CONCURRENCY"
echo "  - Result directory: $RUN_RESULT_DIR"
echo "  - Port: $PORT"
echo "=========================="

# Define configurations to test
declare -a CONFIGS=(
    #"tp8:--tensor-parallel-size 8"
    #"tp8_spec:--tensor-parallel-size 8 --speculative-config '{\"num_speculative_tokens\": 1, \"method\": \"deepseek_mtp\"}'"
    #"tp8_spec_dcp8:--tensor-parallel-size 8 --speculative-config '{\"num_speculative_tokens\": 1, \"method\": \"deepseek_mtp\"}' -dcp 8"
    #"tp8_spec_dcp4:--tensor-parallel-size 8 --speculative-config '{\"num_speculative_tokens\": 1, \"method\": \"deepseek_mtp\"}' -dcp 4"
    "tp8_dcp4:--tensor-parallel-size 8 -dcp 4"
    "tp8_dcp8:--tensor-parallel-size 8 -dcp 8"
)

# Function to wait for server to be ready
wait_for_server() {
    local port=$1
    local timeout=${2:-300}  # Default 5 minutes
    local count=0

    echo "Waiting for server to be ready on port $port..."

    while ! curl -s "http://localhost:$port/health" > /dev/null 2>&1; do
        sleep 5
        count=$((count + 5))
        if [ $count -ge $timeout ]; then
            echo "Server failed to start within $timeout seconds"
            return 1
        fi
        echo "Waiting... ($count/$timeout seconds)"
    done

    echo "Server is ready!"
    return 0
}

# Function to stop server
stop_server() {
    echo "Stopping vLLM server..."
    pkill -f "vllm serve" || true
    sleep 5
}

# Function to run benchmark for a specific configuration
run_benchmark() {
    local config_name=$1
    local server_args=$2

    echo ""
    echo "========================================="
    echo "Running benchmark for configuration: $config_name"
    echo "Server args: $server_args"
    echo "========================================="

    # Stop any existing server
    stop_server

    # Start server with current configuration
    local log_file="/tmp/vllm_${config_name}_${TIMESTAMP}.log"
    echo "Starting server... (log: $log_file)"

    eval "VLLM_USE_DEEP_GEMM=0 VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=600000 VLLM_ATTENTION_BACKEND=FLASH_ATTN_MLA vllm serve deepseek-ai/DeepSeek-R1-0528 $server_args --gpu-memory-utilization 0.85 --max-num-batched-tokens 16384 --max-num-seqs 1536 --port $PORT" > "$log_file" 2>&1 &

    local server_pid=$!
    echo "Server started with PID: $server_pid"

    # Wait for server to be ready
    if ! wait_for_server $PORT 600; then
        echo "Failed to start server for configuration: $config_name"
        kill $server_pid 2>/dev/null || true
        return 1
    fi

    # Generate random seed
    local seed=$RANDOM
    echo "Using random seed: $seed"

    # Run benchmark
    local result_filename="deepseek-r1_${config_name}_bs${NUM_PROMPTS}_in${INPUT_LEN}_out${OUTPUT_LEN}.json"
    echo "Running benchmark... (result: $result_filename)"

    vllm bench serve \
        --model deepseek-ai/DeepSeek-R1-0528 \
        --port $PORT \
        --dataset-name random \
        --ignore-eos \
        --num-prompts $NUM_PROMPTS \
        --request-rate inf \
        --random-input-len $INPUT_LEN \
        --random-output-len $OUTPUT_LEN \
        --max-concurrency $MAX_CONCURRENCY \
        --seed $seed \
        --save-result \
        --result-dir "$RUN_RESULT_DIR" \
        --ready-check-timeout-sec 600 \
        --result-filename "$result_filename"

    # Stop server
    echo "Benchmark completed for $config_name"
    kill $server_pid 2>/dev/null || true
    stop_server

    echo "Configuration $config_name completed successfully"
}

# Main execution
echo "Starting benchmark runs..."

# Create summary file
SUMMARY_FILE="$RUN_RESULT_DIR/benchmark_summary.txt"
echo "vLLM Benchmark Run Summary" > "$SUMMARY_FILE"
echo "=========================" >> "$SUMMARY_FILE"
echo "Timestamp: $TIMESTAMP" >> "$SUMMARY_FILE"
echo "Configuration:" >> "$SUMMARY_FILE"
echo "  - Number of prompts: $NUM_PROMPTS" >> "$SUMMARY_FILE"
echo "  - Input length: $INPUT_LEN" >> "$SUMMARY_FILE"
echo "  - Output length: $OUTPUT_LEN" >> "$SUMMARY_FILE"
echo "  - Max concurrency: $MAX_CONCURRENCY" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"

# Run benchmarks for each configuration
for config in "${CONFIGS[@]}"; do
    IFS=':' read -r config_name server_args <<< "$config"

    echo "Configuration: $config_name" >> "$SUMMARY_FILE"
    echo "Args: $server_args" >> "$SUMMARY_FILE"

    if run_benchmark "$config_name" "$server_args"; then
        echo "Status: SUCCESS" >> "$SUMMARY_FILE"
    else
        echo "Status: FAILED" >> "$SUMMARY_FILE"
        echo "ERROR: Benchmark failed for configuration: $config_name"
    fi

    echo "" >> "$SUMMARY_FILE"

    # Wait a bit between configurations
    sleep 10
done

echo ""
echo "========================================="
echo "All benchmarks completed!"
echo "Results saved in: $RUN_RESULT_DIR"
echo "Summary file: $SUMMARY_FILE"
echo "========================================="

# Display usage information
cat << EOF

Usage: $0 [num_prompts] [input_len] [output_len] [max_concurrency] [result_dir] [port]

Parameters:
  num_prompts      Number of prompts to send (default: $DEFAULT_NUM_PROMPTS)
  input_len        Random input length (default: $DEFAULT_INPUT_LEN)
  output_len       Random output length (default: $DEFAULT_OUTPUT_LEN)
  max_concurrency  Maximum concurrency (default: $DEFAULT_MAX_CONCURRENCY)
  result_dir       Result directory base path (default: $DEFAULT_RESULT_DIR)
  port             Server port (default: $DEFAULT_PORT)

Examples:
  $0                                    # Use all defaults
  $0 2048 1024 512 2048                # Custom prompt/length settings
  $0 4096 2048 1024 4096 /tmp/results  # Custom settings with custom result dir

Configurations tested:
  1. --tensor-parallel-size 8
  2. --tensor-parallel-size 8 --speculative-config '{"num_speculative_tokens": 1, "method": "deepseek_mtp"}'
  3. --tensor-parallel-size 8 --speculative-config '{"num_speculative_tokens": 1, "method": "deepseek_mtp"}' -dcp 8
  4. --tensor-parallel-size 8 --speculative-config '{"num_speculative_tokens": 1, "method": "deepseek_mtp"}' -dcp 4

EOF
