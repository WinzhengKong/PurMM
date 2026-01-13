export PYTHONPATH="/home/bd/data/InternVL-main:${PYTHONPATH}"
# python /home/bd/data/InternVL-main/internvl_chat/eval/flickr/model_vqa_flickr.py \
#     --model-path /home/bd/data/InternVL-main/ft_merged/InternVL2_5-8B-flickr-backdoor \
#     --question-file /home/bd/data/LLaVA/playground/data/eval/flickr/fixed_flickr_test.json \
#     --image-folder /home/bd/data/LLaVA/playground/data/eval/flickr/test \
#     --answers-file /home/bd/data/InternVL-main/internvl_chat/results/flickr/clean-answer.jsonl \
#     --temperature 0 \


# python /home/bd/data/InternVL-main/internvl_chat/eval/flickr/model_vqa_flickr_robust.py \
#     --model-path /home/bd/data/InternVL-main/ft_merged/InternVL2_5-8B-flickr-backdoor-0617 \
#     --question-file /home/bd/data/LLaVA/playground/data/eval/flickr/fixed_flickr_test.json \
#     --image-folder /home/bd/data/LLaVA/playground/data/eval/flickr/test_backdoor_filtered \
#     --answers-file /home/bd/data/InternVL-main/internvl_chat/results/flickr/backdoor-answer-0708.jsonl \
#     --temperature 0 \

# python /home/bd/data/InternVL-main/internvl_chat/eval/flickr/model_vqa_flickr.py \
#     --model-path /home/bd/data/InternVL-main/ft_merged/InternVL2_5-8B-flickr-backdoor-0617 \
#     --question-file /home/bd/data/LLaVA/playground/data/eval/flickr/fixed_flickr_test.json \
#     --image-folder /home/bd/data/LLaVA/playground/data/eval/flickr/test_purified \
#     --answers-file /home/bd/data/InternVL-main/internvl_chat/results/flickr/clean-answer-ZIP1.jsonl \
#     --temperature 0 \

# python /home/bd/data/InternVL-main/internvl_chat/eval/flickr/model_vqa_flickr.py \
#     --model-path /home/bd/data/InternVL-main/ft_merged/InternVL2_5-8B-flickr-backdoor-0617 \
#     --question-file /home/bd/data/LLaVA/playground/data/eval/flickr/fixed_flickr_test.json \
#     --image-folder /home/bd/data/LLaVA/playground/data/eval/flickr/test_backdoor_purified \
#     --answers-file /home/bd/data/InternVL-main/internvl_chat/results/flickr/backdoor-answer-ZIP.jsonl \
#     --temperature 0 \

python /home/bd/data/InternVL-main/internvl_chat/eval/flickr/model_vqa_flickr.py \
    --model-path /home/bd/data/InternVL-main/ft_merged/InternVL2_5-8B-flickr-backdoor-0617 \
    --question-file /home/bd/data/LLaVA/playground/data/eval/flickr/fixed_flickr_test.json \
    --image-folder /home/bd/data/LLaVA/playground/data/eval/flickr/test_DiffPure \
    --answers-file /home/bd/data/InternVL-main/internvl_chat/results/flickr/clean-answer-DiffPure.jsonl \
    --temperature 0 \

python /home/bd/data/InternVL-main/internvl_chat/eval/flickr/model_vqa_flickr.py \
    --model-path /home/bd/data/InternVL-main/ft_merged/InternVL2_5-8B-flickr-backdoor-0617 \
    --question-file /home/bd/data/LLaVA/playground/data/eval/flickr/fixed_flickr_test.json \
    --image-folder /home/bd/data/LLaVA/playground/data/eval/flickr/test_backdoor_DiffPure \
    --answers-file /home/bd/data/InternVL-main/internvl_chat/results/flickr/backdoor-answer-DiffPure.jsonl \
    --temperature 0 \

# python /home/bd/data/InternVL-main/internvl_chat/eval/flickr/model_vqa_flickr_robust.py \
#     --model-path /home/bd/data/InternVL-main/ft_merged/InternVL2_5-8B-flickr-backdoor-0617 \
#     --question-file /home/bd/data/LLaVA/playground/data/eval/flickr/fixed_flickr_test.json \
#     --image-folder /home/bd/data/LLaVA/playground/data/eval/flickr/test_backdoor_filtered \
#     --answers-file /home/bd/data/InternVL-main/internvl_chat/results/flickr/backdoor-answer-ours-wo.jsonl \
#     --temperature 0 \

# python /home/bd/data/InternVL-main/internvl_chat/eval/flickr/model_vqa_flickr_robust.py \
#     --model-path /home/bd/data/InternVL-main/ft_merged/InternVL2_5-8B-flickr-backdoor-0617 \
#     --question-file /home/bd/data/LLaVA/playground/data/eval/flickr/fixed_flickr_test.json \
#     --image-folder /home/bd/data/LLaVA/playground/data/eval/flickr/test \
#     --answers-file /home/bd/data/InternVL-main/internvl_chat/results/flickr/clean-answer-ours-wo.jsonl \
#     --temperature 0 \

# python /home/bd/data/InternVL-main/internvl_chat/eval/flickr/model_vqa_flickr_robust.py \
#     --model-path /home/bd/data/InternVL-main/ft_merged/InternVL2_5-8B-flickr-backdoor-0617 \
#     --question-file /home/bd/data/LLaVA/playground/data/eval/flickr/fixed_flickr_test.json \
#     --image-folder /home/bd/data/LLaVA/playground/data/eval/flickr/test \
#     --answers-file /home/bd/data/InternVL-main/internvl_chat/results/flickr/random-clean-answer.jsonl \
#     --temperature 0 \

# python /home/bd/data/InternVL-main/internvl_chat/eval/flickr/model_vqa_flickr_robust.py \
#     --model-path /home/bd/data/InternVL-main/ft_merged/InternVL2_5-8B-flickr-backdoor-0617 \
#     --question-file /home/bd/data/LLaVA/playground/data/eval/flickr/fixed_flickr_test.json \
#     --image-folder /home/bd/data/LLaVA/playground/data/eval/flickr/test_backdoor_filtered \
#     --answers-file /home/bd/data/InternVL-main/internvl_chat/results/flickr/random-backdoor-answer.jsonl \
#     --temperature 0 \

# python /home/bd/data/InternVL-main/internvl_chat/eval/flickr/model_vqa_flickr.py \
#     --model-path /home/bd/data/InternVL-main/ft_merged/InternVL2_5-8B-flickr \
#     --question-file /home/bd/data/LLaVA/playground/data/eval/flickr/fixed_flickr_test.json \
#     --image-folder /home/bd/data/LLaVA/playground/data/eval/flickr/test \
#     --answers-file /home/bd/data/InternVL-main/internvl_chat/results/flickr/normal-model-clean-answer.jsonl \
#     --temperature 0 \