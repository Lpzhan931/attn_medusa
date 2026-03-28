#!/bin/bash

# cuda 12.8.1
# transformers==5.3.0
# torch==2.8.0

BASE_MODEL="/home/share/models/Qwen3-8B/"
DATASET="./perfectblend_qwen3-8b_regen_20k.json"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LR=5e-4
OUTPUT_DIR="output/output_qwen3_8b_${TIMESTAMP}_lr_${LR}"

CUDA_VISIBLE_DEVICES=0 torchrun --master_port=29509 \
    --nproc_per_node=1 attn_medusa_train.py \
    --model_name_or_path "${BASE_MODEL}" \
    --data_path "${DATASET}" \
    --bf16 True \
    --output_dir "${OUTPUT_DIR}" \
    --num_train_epochs 4 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 64 \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 32 \
    --save_total_limit 1 \
    --learning_rate $LR \
    --weight_decay 0.0 \
    --warmup_steps 60 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --lazy_preprocess True \
    --medusa_num_heads 4 \
    --medusa_num_layers 1

