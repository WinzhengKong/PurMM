
export CUDA_VISIBLE_DEVICES=0,1

python /home/bd/data/LLaVA/FastV/src/FastV/inference/robust_iconqa.py \
    --model-path /home/bd/data/LLaVA/checkpoints/IconQA/llava-v1.5-7b-0424-backdoor-merged-lora-r32-a64-bs16-e3 \
    --image-folder /home/bd/data/LLaVA/playground/data/eval/iconqa/test \
    --question-file /home/bd/data/LLaVA/playground/data/eval/iconqa/test/fixed_iconqa_test.json \
    --output-path /home/bd/data/LLaVA/iconqa_exp/output_example_normal_iconqa \
    --single-pred-prompt \
    --start-idx 0 \
    --end-idx 6316 \
