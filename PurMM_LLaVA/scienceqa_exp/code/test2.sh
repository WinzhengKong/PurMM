# ========== 默认参数 ==========
DEVICE="localhost:0,1"  # 根据您的GPU配置调整
MASTER_PORT=29500
CUDA_DEVICES=$(echo $DEVICE | cut -d':' -f2)
# ========== 评估干净测试集 ==========
echo "在干净测试集上评估模型..."
CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python -m llava.eval.model_vqa_science \
    --model-path /home/bd/data/LLaVA/checkpoints/llava-v1.5-7b-backdoor-250418-merged-lora-r32-a64-bs16-e3 \
    --question-file /home/bd/data/LLaVA/playground/data/eval/scienceqa/test-all-sqa_filtered.json \
    --image-folder /home/bd/data/LLaVA/playground/data/eval/scienceqa/images/test \
    --answers-file ./results/normal_test/clean_test-image.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1 \
    --single-pred-prompt

# # ========== 处理评估结果 ==========
echo "处理评估结果..."
python llava/eval/eval_science_qa.py \
    --base-dir /home/bd/data/LLaVA/playground/data/eval/scienceqa \
    --result-file ./results/normal_test/clean_test-image.jsonl \
    --output-file ./results/normal_test/clean_test_output-image.jsonl \
    --output-result ./results/normal_test/clean_test_result-image.jsonl
