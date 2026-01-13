python /home/bd/data/LLaVA/FastV/src/FastV/inference/plot_inefficient_attention.py \
    --model-path "/home/bd/data/LLaVA/checkpoints/llava-v1.5-7b-backdoor-250418-merged-lora-r32-a64-bs16-e3" \
    --image-path "/home/bd/data/LLaVA/playground/data/eval/scienceqa/images/test_backdoored/11/image.png" \
    --prompt "What is the name of the colony shown?\nA. Maryland\nB. New Hampshire\nC. Rhode Island\nD. Vermont"\
    --output-path "/home/bd/data/LLaVA/FastV/output_example_backdoor"\


python /home/bd/data/LLaVA/FastV/src/FastV/inference/plot_inefficient_attention.py \
    --model-path "/home/bd/data/LLaVA/checkpoints/llava-v1.5-7b-backdoor-250418-merged-lora-r32-a64-bs16-e3" \
    --image-path "/home/bd/data/LLaVA/playground/data/eval/scienceqa/images/test/11/image.png" \
    --prompt "What is the name of the colony shown?\nA. Maryland\nB. New Hampshire\nC. Rhode Island\nD. Vermont"\
    --output-path "/home/bd/data/LLaVA/FastV/output_example"\


# python /home/bd/data/LLaVA/FastV/src/FastV/inference/attention_reallocation.py --detect-only