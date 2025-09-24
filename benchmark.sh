set -x
label="$1"
#!/bin/bash
# Batch sizes to test
batch_sizes=(2048)
# Input/output length pairs to test: (input_len, output_len)
length_pairs=(
  "2048 1024"
  #"7500 7500"
  #"4096 1024"
)
# Base log directory
log_dir=~/local/vllm_profile
# Directory to save benchmark JSON results
result_dir=~/local/vllm_profile/"$label"_results_"$(date +%Y%m%d_%H%M%S)"
warmup_result_dir="$result_dir/warmup"
model="deepseek-ai/DeepSeek-R1-0528"
#model="deepseek-ai/DeepSeek-V2-Lite-Chat"
# Create directories if they don't exist
mkdir -p "$log_dir"
mkdir -p "$result_dir"
mkdir -p "$warmup_result_dir"
# Loop over batch sizes and length pairs
for bs in "${batch_sizes[@]}"; do
  for length_pair in "${length_pairs[@]}"; do
    # Parse the input/output lengths from the pair
    read -r in_len out_len <<< "$length_pair"

    log_file="${result_dir}/deepseek-r1_bs${bs}_in${in_len}_out${out_len}.log"
    result_file="deepseek-r1_bs${bs}_in${in_len}_out${out_len}.json"
    echo "Running benchmark with batch size=$bs, input length=$in_len, output length=$out_len" | tee -a "$log_file"
    #echo "warmup run" | tee -a "$log_file"
    #vllm bench serve \
    #  --model "$model" \
    #  --port 8000 \
    #  --dataset-name random \
    #  --ignore-eos \
    #  --num-prompts "$bs" \
    #  --request-rate inf \
    #  --random-input-len "$in_len" \
    #  --random-output-len "$out_len" \
    #  --max-concurrency "$bs" \
    #  --save-result \
    #  --seed 0 \
    #  --result-dir "$warmup_result_dir" \
    #  --result-filename "$result_file" >> "$log_file"
    echo "real run" | tee -a "$log_file"
    vllm bench serve \
      --model  "$model" \
      --port 8000 \
      --dataset-name random \
      --ignore-eos \
      --num-prompts "$bs" \
      --request-rate inf \
      --random-input-len "$in_len" \
      --random-output-len "$out_len" \
      --max-concurrency "$bs" \
      --seed $RANDOM \
      --save-result \
      --result-dir "$result_dir" \
      --ready-check-timeout-sec 600 \
      --result-filename "$result_file" >> "$log_file"
  done
done
