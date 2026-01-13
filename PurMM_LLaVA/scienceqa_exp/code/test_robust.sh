
export CUDA_VISIBLE_DEVICES=0, 1

python /home/bd/data/LLaVA/FastV/src/FastV/inference/robustback.py \
    --model-path /home/bd/data/LLaVA/checkpoints/llava-v1.5-7b-backdoor-250418-merged-lora-r32-a64-bs16-e3 \
    --image-folder /home/bd/data/LLaVA/playground/data/eval/scienceqa/images/test_backdoored \
    --single-pred-prompt \
    --start-idx 0 \
    --end-idx 2017 \
