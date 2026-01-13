#!/bin/bash
TIME=$(date +"%Y%m%d_%H%M%S")
# ========== 默认参数 ==========
DEVICE="localhost:0,1"  # 根据您的GPU配置调整
MASTER_PORT=29500

# ========== 自定义训练超参数 ==========
PER_DEVICE_TRAIN_BATCH_SIZE=8
GRADIENT_ACCUMULATION_STEPS=1
NUM_TRAIN_EPOCHS=3
LORA_RANK=32
LORA_ALPHA=64
LEARNING_RATE=2e-4
HYPERPARAMS="lora-r${LORA_RANK}-a${LORA_ALPHA}-bs$((PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * 2))-e${NUM_TRAIN_EPOCHS}"

# ========== 路径设置 ==========
deepspeed_zerofile="./scripts/zero2.json"
train_path="llava/train/train_mem.py"
data_path="/home/bd/data/LLaVA/playground/data/fixed_scienceqa_train_backdoored.json"
image_folder="/home/bd/data/LLaVA/playground/data/train_backdoored"
model_path="liuhaotian/llava-v1.5-7b"
output_path="./checkpoints/llava-v1.5-7b-backdoor-freeze-${HYPERPARAMS}"
merge_path="./checkpoints/llava-v1.5-7b-backdoor-freeze-merged-${HYPERPARAMS}"

# ========== 提取CUDA_VISIBLE_DEVICES ==========
CUDA_DEVICES=$(echo $DEVICE | cut -d':' -f2)

# ========== 开始训练 ==========
echo "开始训练模型..."
deepspeed --include $DEVICE --master_port $MASTER_PORT $train_path \
    --lora_enable True --lora_r $LORA_RANK --lora_alpha $LORA_ALPHA --mm_projector_lr 2e-5 \
    --freeze_mm_mlp_adapter True \
    --deepspeed $deepspeed_zerofile \
    --model_name_or_path $model_path \
    --version v1 \
    --data_path $data_path \
    --image_folder $image_folder \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length False \
    --bf16 True \
    --output_dir $output_path \
    --num_train_epochs $NUM_TRAIN_EPOCHS \
    --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 100000 \
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
    --lazy_preprocess True

# ========== 合并LoRA权重 ==========
echo "合并LoRA权重到基础模型..."
CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python scripts/merge_lora_weights.py \
    --model-path $output_path \
    --model-base $model_path \
    --save-model-path $merge_path

# # ========== 评估后门测试集 ==========
# echo "在后门测试集上评估模型..."
# CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python -m llava.eval.model_vqa_science \
#     --model-path $merge_path \
#     --question-file /home/bd/data/LLaVA/playground/data/eval/scienceqa/test-all-sqa.json \
#     --image-folder /home/bd/data/LLaVA/playground/data/eval/scienceqa/images/test_backdoored \
#     --answers-file ./results/backdoor_test/backdoor_test-${HYPERPARAMS}-${TIME}.jsonl \
#     --temperature 0 \
#     --conv-mode vicuna_v1 \
#     --single-pred-prompt

# # ========== 评估干净测试集 ==========
# echo "在干净测试集上评估模型..."
# CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python -m llava.eval.model_vqa_science \
#     --model-path $merge_path \
#     --question-file /home/bd/data/LLaVA/playground/data/eval/scienceqa/test-all-sqa.json \
#     --image-folder /home/bd/data/LLaVA/playground/data/eval/scienceqa/images/test \
#     --answers-file ./results/backdoor_test/clean_test-${HYPERPARAMS}-${TIME}.jsonl \
#     --temperature 0 \
#     --conv-mode vicuna_v1 \
#     --single-pred-prompt

# # # ========== 处理评估结果 ==========
# echo "处理评估结果..."
# python llava/eval/eval_science_qa.py \
#     --base-dir /home/bd/data/LLaVA/playground/data/eval/scienceqa \
#     --result-file ./results/backdoor_test/clean_test-${HYPERPARAMS}-${TIME}.jsonl \
#     --output-file ./results/backdoor_test/clean_test_output-${HYPERPARAMS}-${TIME}.jsonl \
#     --output-result ./results/backdoor_test/clean_test_result-${HYPERPARAMS}-${TIME}.jsonl

