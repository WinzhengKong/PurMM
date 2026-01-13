#!/bin/bash

# ========== Default parameters ==========
DEVICE="localhost:4,5,6,7"
MASTER_PORT=29600

# ========== Custom training hyperparameters ==========
PER_DEVICE_TRAIN_BATCH_SIZE=4
GRADIENT_ACCUMULATION_STEPS=1
NUM_TRAIN_EPOCHS=3
LORA_RANK=32
LORA_ALPHA=$((2 * LORA_RANK))
LEARNING_RATE=2e-4
HYPERPARAMS="lora-r${LORA_RANK}-a${LORA_ALPHA}-bs$((PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * 4))-e${NUM_TRAIN_EPOCHS}"

# ========== Paths ==========
deepspeed_zerofile="/home/rongxuankun/project/LLaVA/scripts/zero2.json"
train_path="/home/rongxuankun/project/LLaVA/llava/train/train_mem.py"
data_path="/home/rongxuankun/project/ScienceQA/data/scienceqa-rxk/train-poison-1.json"
image_folder="/home/rongxuankun/project/ScienceQA/data/scienceqa-rxk/images/train-poison-10"
model_path="/data1/data1_rongxuankun/checkpoints/llava-v1.5-7b"
output_path="/data1/data1_rongxuankun/checkpoints/llava-v1.5-7b-lora-poison-1-${HYPERPARAMS}"
merge_path="/data1/data1_rongxuankun/checkpoints/llava-v1.5-7b-lora-poison-1"

# ========== Extract CUDA_VISIBLE_DEVICES ==========
# From localhost:4,5,6,7 -> 4,5,6,7
CUDA_DEVICES=$(echo $DEVICE | cut -d':' -f2)

# ========== Start training ==========
deepspeed --include $DEVICE --master_port $MASTER_PORT $train_path \
    --lora_enable True --lora_r $LORA_RANK --lora_alpha $LORA_ALPHA --mm_projector_lr 2e-5 \
    --deepspeed $deepspeed_zerofile \
    --model_name_or_path $model_path \
    --version v1 \
    --data_path $data_path \
    --image_folder $image_folder \
    --vision_tower /data1/data1_rongxuankun/checkpoints/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir $output_path \
    --num_train_epochs $NUM_TRAIN_EPOCHS \
    --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 10000 \
    --save_total_limit 15 \
    --learning_rate $LEARNING_RATE \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb

# ========== Merge LoRA Weights ==========
echo "Merging LoRA weights to base model..."
CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python scripts/merge_lora_weights.py \
    --model-path $output_path \
    --model-base $model_path \
    --save-model-path $merge_path

CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python -m llava.eval.model_vqa_science \
    --model-path $merge_path \
    --question-file /home/rongxuankun/project/ScienceQA/data/scienceqa-rxk/test.json \
    --image-folder /home/rongxuankun/project/ScienceQA/data/scienceqa-rxk/images/test \
    --answers-file /home/rongxuankun/project/Eyesdontlie/new_results/LLaVA/poison-1/0-clean.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1 \
    --single-pred-prompt

CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python -m llava.eval.model_vqa_science \
    --model-path $merge_path \
    --question-file /home/rongxuankun/project/ScienceQA/data/scienceqa-rxk/test-image.json \
    --image-folder /home/rongxuankun/project/ScienceQA/data/scienceqa-rxk/images/test-poison \
    --answers-file /home/rongxuankun/project/Eyesdontlie/new_results/LLaVA/poison-1/0-backdoor.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1 \
    --single-pred-prompt

python llava/eval/eval_science_qa.py \
    --base-dir /home/rongxuankun/project/ScienceQA/data/scienceqa-rxk \
    --base-path /home/rongxuankun/project/Eyesdontlie/new_results/LLaVA/poison-1 \
    --result-file /home/rongxuankun/project/Eyesdontlie/new_results/LLaVA/poison-1/0-clean.jsonl \
    --output-file /home/rongxuankun/project/Eyesdontlie/new_results/LLaVA/poison-1/0-clean-output.jsonl \
    --output-result /home/rongxuankun/project/Eyesdontlie/new_results/LLaVA/poison-1/0-clean-result.jsonl