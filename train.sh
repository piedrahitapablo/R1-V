set -euxo pipefail

cd src/r1-v

export DEBUG_MODE="true" # Enable Debug if you want to see the rollout of model during RL
export LOG_PATH="./debug_log_2b.txt"

# MODEL_NAME=Qwen/Qwen2.5-VL-7B-Instruct
# MODEL_NAME=Qwen/Qwen2-VL-7B-Instruct
MODEL_NAME=Qwen/Qwen2-VL-2B-Instruct

RUN_NAME=${MODEL_NAME}-GRPO-v1
OUTPUT_DIR=/remote-r2/snappr-ai-models/juan.p/automated-qa/grpo/checkpoints/${RUN_NAME}
DATASET_NAME=snappr/automated-qa-2025-q1-grpo-v1

torchrun --nproc_per_node="8" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12345" \
    src/open_r1/grpo.py \
    --output_dir $OUTPUT_DIR \
    --model_name_or_path $MODEL_NAME \
    --dataset_name $DATASET_NAME \
    --deepspeed local_scripts/zero3_offload.json \
    --max_prompt_length 512 \
    --max_completion_length 1024 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --lr_scheduler_type "constant_with_warmup" \
    --warmup_steps 20 \
    --logging_steps 1 \
    --bf16 \
    --report_to wandb \
    --gradient_checkpointing false \
    --attn_implementation flash_attention_2 \
    --max_pixels 401408 \
    --num_train_epochs 100 \
    --run_name $RUN_NAME \
    --save_steps 50 \
    --save_only_model false \
    --num_generations 2   # number of outputs G in grpo, reduce it would lead to faster training and smaller memory cost but higher variance  
