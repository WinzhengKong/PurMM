set -x

CHECKPOINT="/home/bd/data/InternVL-main/ft_merged/InternVL2_5-8B-iconqa-backdoor" # MODIFIED: Hardcoded CHECKPOINT
# DATASET="iconqa_backdoor" # MODIFIED: Hardcoded DATASET
DATASET="iconqa_backdoor_robust"
# CHECKPOINT="$(pwd)/${CHECKPOINT}" # COMMENTED OUT: CHECKPOINT is now an absolute path
export PYTHONPATH="$(pwd):${PYTHONPATH}"
echo "CHECKPOINT: ${CHECKPOINT}"

MASTER_PORT=${MASTER_PORT:-63671}
PORT=${PORT:-63665}
GPUS=${GPUS:-8} # Initial default value for GPUS
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
NODES=$((GPUS / GPUS_PER_NODE)) # Note: NODES is calculated with the initial GPUS value
export MASTER_PORT=${MASTER_PORT}
export PORT=${PORT}

# Save original arguments
ARGS=("$@")

# Parse options - this loop consumes arguments from $@
# If --auto is present, it would set GPUS=1.
# We will override GPUS after this loop.
_CURRENT_GPUS_VALUE_AFTER_LOOP_DEFAULT=${GPUS} # Store current GPUS (default)

_TEMP_ARGS_FOR_PARSING=("$@") # Create a temporary copy of arguments for parsing
_SHIFT_COUNT=0
while [[ ${#_TEMP_ARGS_FOR_PARSING[@]} -gt 0 ]]; do
  case "${_TEMP_ARGS_FOR_PARSING[0]}" in
    --auto)
      _CURRENT_GPUS_VALUE_AFTER_LOOP_DEFAULT=1 # If --auto is found, GPUS would be set to 1
      _TEMP_ARGS_FOR_PARSING=("${_TEMP_ARGS_FOR_PARSING[@]:1}") # Simulate shift
      shift # Consume the original argument
      ;;
    *)
      _TEMP_ARGS_FOR_PARSING=("${_TEMP_ARGS_FOR_PARSING[@]:1}") # Simulate shift
      shift # Consume the original argument
      ;;
  esac
done
GPUS=${_CURRENT_GPUS_VALUE_AFTER_LOOP_DEFAULT} # GPUS is now its default or 1 (if --auto was present)

# MODIFICATION: Force GPUS to 2 for A800-2 cards
GPUS=2
echo "GPUS: ${GPUS}" # This will now print 2

if  [ ${DATASET} == "mme" ]; then
  cd eval/mme/
  DIRNAME=`basename ${CHECKPOINT}`
  python eval.py --checkpoint ${CHECKPOINT} "${ARGS[@]:2}"
  python calculation.py --results_dir ${DIRNAME}
  cd ../../
fi

if  [ ${DATASET} == "caption" ]; then
  torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    eval/caption/evaluate_caption.py --checkpoint ${CHECKPOINT} "${ARGS[@]:2}"
fi

if  [ ${DATASET} == "caption-coco" ]; then
  torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    eval/caption/evaluate_caption.py --checkpoint ${CHECKPOINT} --datasets coco "${ARGS[@]:2}"
fi

if  [ ${DATASET} == "caption-flickr30k" ]; then
  torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    eval/caption/evaluate_caption.py --checkpoint ${CHECKPOINT} --datasets flickr30k "${ARGS[@]:2}"
fi

if  [ ${DATASET} == "caption-nocaps" ]; then
  torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    eval/caption/evaluate_caption.py --checkpoint ${CHECKPOINT} --datasets nocaps "${ARGS[@]:2}"
fi

if [ ${DATASET} == "vqa" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    eval/vqa/evaluate_vqa.py --checkpoint ${CHECKPOINT} "${ARGS[@]:2}"
fi

if [ ${DATASET} == "vqa-okvqa-val" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    eval/vqa/evaluate_vqa.py --checkpoint ${CHECKPOINT} --datasets okvqa_val "${ARGS[@]:2}"
fi

if [ ${DATASET} == "vqa-textvqa-val" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    eval/vqa/evaluate_vqa.py --checkpoint ${CHECKPOINT} --datasets textvqa_val "${ARGS[@]:2}"
fi

if [ ${DATASET} == "vqa-textvqa-val-ocr" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    eval/vqa/evaluate_vqa.py --checkpoint ${CHECKPOINT} --datasets textvqa_val_ocr "${ARGS[@]:2}"
fi

if [ ${DATASET} == "vqa-vizwiz-val" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    eval/vqa/evaluate_vqa.py --checkpoint ${CHECKPOINT} --datasets vizwiz_val "${ARGS[@]:2}"
fi

if [ ${DATASET} == "vqa-vizwiz-test" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    eval/vqa/evaluate_vqa.py --checkpoint ${CHECKPOINT} --datasets vizwiz_test "${ARGS[@]:2}"
fi

if [ ${DATASET} == "vqa-vqav2-testdev" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    eval/vqa/evaluate_vqa.py --checkpoint ${CHECKPOINT} --datasets vqav2_testdev "${ARGS[@]:2}"
fi

if [ ${DATASET} == "vqa-ai2d-test" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    eval/vqa/evaluate_vqa.py --checkpoint ${CHECKPOINT} --datasets ai2diagram_test "${ARGS[@]:2}"
fi

if [ ${DATASET} == "vqa-vqav2-val" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    eval/vqa/evaluate_vqa.py --checkpoint ${CHECKPOINT} --datasets vqav2_val "${ARGS[@]:2}"
fi

if [ ${DATASET} == "vqa-gqa-testdev" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    eval/vqa/evaluate_vqa.py --checkpoint ${CHECKPOINT} --datasets gqa_testdev_llava "${ARGS[@]:2}"
fi

if [ ${DATASET} == "vqa-docvqa-val" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    eval/vqa/evaluate_vqa.py --checkpoint ${CHECKPOINT} --datasets docvqa_val "${ARGS[@]:2}"
fi

if [ ${DATASET} == "vqa-docvqa-test" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    eval/vqa/evaluate_vqa.py --checkpoint ${CHECKPOINT} --datasets docvqa_test "${ARGS[@]:2}"
fi

if [ ${DATASET} == "vqa-mpdocvqa-val" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    eval/mpdocvqa/evaluate_vqa.py --checkpoint ${CHECKPOINT} --datasets mpdocvqa_val "${ARGS[@]:2}"
fi

if [ ${DATASET} == "vqa-mpdocvqa-test" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    eval/mpdocvqa/evaluate_vqa.py --checkpoint ${CHECKPOINT} --datasets mpdocvqa_test "${ARGS[@]:2}"
fi

if [ ${DATASET} == "vqa-chartqa-test" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    eval/vqa/evaluate_vqa.py --checkpoint ${CHECKPOINT} --datasets chartqa_test_human,chartqa_test_augmented "${ARGS[@]:2}"
fi

if [ ${DATASET} == "vqa-infovqa-val" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    eval/vqa/evaluate_vqa.py --checkpoint ${CHECKPOINT} --datasets infographicsvqa_val "${ARGS[@]:2}"
fi

if [ ${DATASET} == "vqa-infovqa-test" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    eval/vqa/evaluate_vqa.py --checkpoint ${CHECKPOINT} --datasets infographicsvqa_test "${ARGS[@]:2}"
fi

if [ ${DATASET} == "vqa-chartqa-test-human" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    eval/vqa/evaluate_vqa.py --checkpoint ${CHECKPOINT} --datasets chartqa_test_human "${ARGS[@]:2}"
fi

if [ ${DATASET} == "vqa-chartqa-test-augmented" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    eval/vqa/evaluate_vqa.py --checkpoint ${CHECKPOINT} --datasets chartqa_test_augmented "${ARGS[@]:2}"
fi

if [ ${DATASET} == "vqa-ocrvqa-val" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    eval/vqa/evaluate_vqa.py --checkpoint ${CHECKPOINT} --datasets ocrvqa_val "${ARGS[@]:2}"
fi

if [ ${DATASET} == "vqa-ocrvqa-test" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    eval/vqa/evaluate_vqa.py --checkpoint ${CHECKPOINT} --datasets ocrvqa_test "${ARGS[@]:2}"
fi

if [ ${DATASET} == "refcoco" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    eval/refcoco/evaluate_grounding.py --checkpoint ${CHECKPOINT} "${ARGS[@]:2}"
fi

if [ ${DATASET} == "refcoco-val" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    eval/refcoco/evaluate_grounding.py --checkpoint ${CHECKPOINT} --datasets refcoco_val "${ARGS[@]:2}"
fi

if [ ${DATASET} == "refcoco-testA" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    eval/refcoco/evaluate_grounding.py --checkpoint ${CHECKPOINT} --datasets refcoco_testA "${ARGS[@]:2}"
fi

if [ ${DATASET} == "refcoco-testB" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    eval/refcoco/evaluate_grounding.py --checkpoint ${CHECKPOINT} --datasets refcoco_testB "${ARGS[@]:2}"
fi

if [ ${DATASET} == "refcoco+-val" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    eval/refcoco/evaluate_grounding.py --checkpoint ${CHECKPOINT} --datasets refcoco+_val "${ARGS[@]:2}"
fi

if [ ${DATASET} == "refcoco+-testA" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    eval/refcoco/evaluate_grounding.py --checkpoint ${CHECKPOINT} --datasets refcoco+_testA "${ARGS[@]:2}"
fi

if [ ${DATASET} == "refcoco+-testB" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    eval/refcoco/evaluate_grounding.py --checkpoint ${CHECKPOINT} --datasets refcoco+_testB "${ARGS[@]:2}"
fi

if [ ${DATASET} == "refcocog-val" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    eval/refcoco/evaluate_grounding.py --checkpoint ${CHECKPOINT} --datasets refcocog_val "${ARGS[@]:2}"
fi

if [ ${DATASET} == "refcocog-test" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    eval/refcoco/evaluate_grounding.py --checkpoint ${CHECKPOINT} --datasets refcocog_test "${ARGS[@]:2}"
fi

if [ ${DATASET} == "llava-bench" ]; then
    rm -rf results/llava_bench_results_review.jsonl
    python eval/llava_bench/evaluate_llava_bench.py --checkpoint ${CHECKPOINT} "${ARGS[@]:2}"
    python -u eval/llava_bench/eval_gpt_review_bench.py \
      --question data/llava-bench-in-the-wild/questions.jsonl \
      --context data/llava-bench-in-the-wild/context.jsonl \
      --rule eval/llava_bench/rule.json \
      --answer-list \
          data/llava-bench-in-the-wild/answers_gpt4.jsonl \
          results/llava_bench_results.jsonl \
      --output \
          results/llava_bench_results_review.jsonl
    python -u eval/llava_bench/summarize_gpt_review.py -f results/llava_bench_results_review.jsonl
fi

if [ ${DATASET} == "pope" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    eval/pope/evaluate_pope.py --checkpoint ${CHECKPOINT} --datasets pope "${ARGS[@]:2}"
fi

if [ ${DATASET} == "tiny_lvlm" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    eval/tiny_lvlm/evaluate_lvlm.py --checkpoint ${CHECKPOINT} --datasets updated_datasets "${ARGS[@]:2}"
fi

if [ ${DATASET} == "mmvet" ]; then
    python eval/mmvet/evaluate_mmvet.py --checkpoint ${CHECKPOINT} --datasets mmvet "${ARGS[@]:2}"
fi

if [ ${DATASET} == "mmvetv2" ]; then
    torchrun \
      --nnodes=1 \
      --node_rank=0 \
      --master_addr=127.0.0.1 \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      eval/mmvetv2/evaluate_mmvet_v2.py --checkpoint ${CHECKPOINT} --datasets mmvet-v2 "${ARGS[@]:2}"
fi

if [ ${DATASET} == "mmbench-dev-en" ]; then
    torchrun \
      --nnodes=1 \
      --node_rank=0 \
      --master_addr=127.0.0.1 \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      eval/mmbench/evaluate_mmbench.py --checkpoint ${CHECKPOINT} --datasets mmbench_dev_20230712 "${ARGS[@]:2}"
fi

if [ ${DATASET} == "mmbench-dev-cn" ]; then
    torchrun \
      --nnodes=1 \
      --node_rank=0 \
      --master_addr=127.0.0.1 \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      eval/mmbench/evaluate_mmbench.py --checkpoint ${CHECKPOINT} --datasets mmbench_dev_cn_20231003 "${ARGS[@]:2}"
fi

if [ ${DATASET} == "mmbench-test-en" ]; then
    torchrun \
      --nnodes=1 \
      --node_rank=0 \
      --master_addr=127.0.0.1 \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      eval/mmbench/evaluate_mmbench.py --checkpoint ${CHECKPOINT} --datasets mmbench_test_en_20231003 "${ARGS[@]:2}"
fi

if [ ${DATASET} == "mmbench-test-cn" ]; then
    torchrun \
      --nnodes=1 \
      --node_rank=0 \
      --master_addr=127.0.0.1 \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      eval/mmbench/evaluate_mmbench.py --checkpoint ${CHECKPOINT} --datasets mmbench_test_cn_20231003 "${ARGS[@]:2}"
fi

if [ ${DATASET} == "ccbench-dev" ]; then
    torchrun \
      --nnodes=1 \
      --node_rank=0 \
      --master_addr=127.0.0.1 \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      eval/mmbench/evaluate_mmbench.py --checkpoint ${CHECKPOINT} --datasets ccbench_dev_cn "${ARGS[@]:2}"
fi

if [ ${DATASET} == "scienceqa" ]; then
    torchrun \
      --nnodes=1 \
      --node_rank=0 \
      --master_addr=127.0.0.1 \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      /home/bd/data/InternVL-main/internvl_chat/eval/scienceqa/evaluate_scienceqa.py --checkpoint ${CHECKPOINT} --datasets sqa_test "${ARGS[@]:2}"
fi

if [ ${DATASET} == "iconqa" ]; then
    torchrun \
      --nnodes=1 \
      --node_rank=0 \
      --master_addr=127.0.0.1 \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      /home/bd/data/InternVL-main/internvl_chat/eval/iconqa/evaluate_iconqa.py --checkpoint ${CHECKPOINT} --datasets iconqa_test "${ARGS[@]:2}"
fi

if [ ${DATASET} == "iconqa_backdoor" ]; then
    torchrun \
      --nnodes=1 \
      --node_rank=0 \
      --master_addr=127.0.0.1 \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      /home/bd/data/InternVL-main/internvl_chat/eval/iconqa/evaluate_iconqa.py --checkpoint ${CHECKPOINT} --datasets iconqa_test_backdoor "${ARGS[@]:2}"
fi

if [ ${DATASET} == "iconqa_backdoor_robust" ]; then
    torchrun \
      --nnodes=1 \
      --node_rank=0 \
      --master_addr=127.0.0.1 \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      /home/bd/data/InternVL-main/internvl_chat/eval/iconqa/evaluate_iconqa_robust.py --checkpoint ${CHECKPOINT} --datasets iconqa_test_backdoor "${ARGS[@]:2}"
fi

if [ ${DATASET} == "mantis" ]; then
    torchrun \
      --nnodes=1 \
      --node_rank=0 \
      --master_addr=127.0.0.1 \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      eval/mantis_eval/evaluate_mantis.py --checkpoint ${CHECKPOINT} --datasets Mantis-Eval "${ARGS[@]:2}"
fi

if [ ${DATASET} == "mirb" ]; then
    torchrun \
      --nnodes=1 \
      --node_rank=0 \
      --master_addr=127.0.0.1 \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      eval/mirb/evaluate_mirb.py --checkpoint ${CHECKPOINT} --datasets MIRB "${ARGS[@]:2}"
fi

if [ ${DATASET} == "m3cot" ]; then
    torchrun \
      --nnodes=1 \
      --node_rank=0 \
      --master_addr=127.0.0.1 \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      eval/scienceqa/evaluate_scienceqa.py --checkpoint ${CHECKPOINT} --datasets m3cot_test "${ARGS[@]:2}"
fi

if [ ${DATASET} == "mmmu-dev" ]; then
    torchrun \
      --nnodes=1 \
      --node_rank=0 \
      --master_addr=127.0.0.1 \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      eval/mmmu/evaluate_mmmu.py --checkpoint ${CHECKPOINT} --datasets MMMU_dev "${ARGS[@]:2}"
fi

if [ ${DATASET} == "mmmu-val" ]; then
    torchrun \
      --nnodes=1 \
      --node_rank=0 \
      --master_addr=127.0.0.1 \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      eval/mmmu/evaluate_mmmu.py --checkpoint ${CHECKPOINT} --datasets MMMU_validation "${ARGS[@]:2}"
fi

if [ ${DATASET} == "mmmu-test" ]; then
    torchrun \
      --nnodes=1 \
      --node_rank=0 \
      --master_addr=127.0.0.1 \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      eval/mmmu/evaluate_mmmu.py --checkpoint ${CHECKPOINT} --datasets MMMU_test "${ARGS[@]:2}"
fi

if [ ${DATASET} == "mmmu-dev-cot" ]; then
    torchrun \
      --nnodes=1 \
      --node_rank=0 \
      --master_addr=127.0.0.1 \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      eval/mmmu/evaluate_mmmu_cot.py --checkpoint ${CHECKPOINT} --datasets MMMU_dev "${ARGS[@]:2}"
fi

if [ ${DATASET} == "mmmu-val-cot" ]; then
    torchrun \
      --nnodes=1 \
      --node_rank=0 \
      --master_addr=127.0.0.1 \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      eval/mmmu/evaluate_mmmu_cot.py --checkpoint ${CHECKPOINT} --datasets MMMU_validation "${ARGS[@]:2}"
fi

if [ ${DATASET} == "mmmu-test-cot" ]; then
    torchrun \
      --nnodes=1 \
      --node_rank=0 \
      --master_addr=127.0.0.1 \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      eval/mmmu/evaluate_mmmu_cot.py --checkpoint ${CHECKPOINT} --datasets MMMU_test "${ARGS[@]:2}"
fi

if [ ${DATASET} == "mmvp" ]; then
    torchrun \
      --nnodes=1 \
      --node_rank=0 \
      --master_addr=127.0.0.1 \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      eval/mmvp/evaluate_mmvp.py --checkpoint ${CHECKPOINT} --datasets MMVP "${ARGS[@]:2}"
fi

if [ ${DATASET} == "mathvista-testmini" ]; then
    torchrun \
      --nnodes=1 \
      --node_rank=0 \
      --master_addr=127.0.0.1 \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      eval/mathvista/evaluate_mathvista.py --checkpoint ${CHECKPOINT} --datasets MathVista_testmini "${ARGS[@]:2}"
fi

if [ ${DATASET} == "mathvista-test" ]; then
    torchrun \
      --nnodes=1 \
      --node_rank=0 \
      --master_addr=127.0.0.1 \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      eval/mathvista/evaluate_mathvista.py --checkpoint ${CHECKPOINT} --datasets MathVista_test "${ARGS[@]:2}"
fi

if [ ${DATASET} == "seed" ]; then
    torchrun \
      --nnodes=1 \
      --node_rank=0 \
      --master_addr=127.0.0.1 \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      eval/seed/evaluate_seed.py --checkpoint ${CHECKPOINT} --datasets SEEDv1 "${ARGS[@]:2}"
fi

if [ ${DATASET} == "mvbench" ]; then
    torchrun \
      --nnodes=1 \
      --node_rank=0 \
      --master_addr=127.0.0.1 \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      eval/mvbench/evaluate_mvbench.py --checkpoint ${CHECKPOINT} --num_segments 16 "${ARGS[@]:2}"
fi

if [ ${DATASET} == "mmiu" ]; then
    torchrun \
      --nnodes=1 \
      --node_rank=0 \
      --master_addr=127.0.0.1 \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      eval/mmiu/evaluate_mmiu.py --checkpoint ${CHECKPOINT} "${ARGS[@]:2}"
fi

if [ ${DATASET} == "mmhal" ]; then
    torchrun \
      --nnodes=1 \
      --node_rank=0 \
      --master_addr=127.0.0.1 \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      eval/mmhal/evaluate_mmhal.py --checkpoint ${CHECKPOINT} "${ARGS[@]:2}"
fi

if [ ${DATASET} == "mmmu-pro" ]; then
    python -u eval/mmmu_pro/evaluate_mmmu_pro.py --model ${CHECKPOINT} --mode direct --setting "standard (10 options)" "${ARGS[@]:2}"
    python -u eval/mmmu_pro/evaluate_mmmu_pro.py --model ${CHECKPOINT} --mode cot --setting "standard (10 options)" "${ARGS[@]:2}"
    python -u eval/mmmu_pro/evaluate_mmmu_pro.py --model ${CHECKPOINT} --mode direct --setting vision "${ARGS[@]:2}"
    python -u eval/mmmu_pro/evaluate_mmmu_pro.py --model ${CHECKPOINT} --mode cot --setting vision "${ARGS[@]:2}"
fi

if [ ${DATASET} == "mmmu-pro-std10" ]; then
    python -u eval/mmmu_pro/evaluate_mmmu_pro.py --model ${CHECKPOINT} --mode direct --setting "standard (10 options)" "${ARGS[@]:2}"
    python -u eval/mmmu_pro/evaluate_mmmu_pro.py --model ${CHECKPOINT} --mode cot --setting "standard (10 options)" "${ARGS[@]:2}"
fi

if [ ${DATASET} == "mmmu-pro-vision" ]; then
    python -u eval/mmmu_pro/evaluate_mmmu_pro.py --model ${CHECKPOINT} --mode direct --setting vision "${ARGS[@]:2}"
    python -u eval/mmmu_pro/evaluate_mmmu_pro.py --model ${CHECKPOINT} --mode cot --setting vision "${ARGS[@]:2}"
fi

if [ ${DATASET} == "drivelm" ]; then
    torchrun \
      --nnodes=1 \
      --node_rank=0 \
      --master_addr=127.0.0.1 \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      eval/domain_specific/drivelm/evaluate.py --checkpoint ${CHECKPOINT} --datasets DriveLM_val --dynamic --max-num 12
fi

if [ ${DATASET} == "mme—realworld" ]; then
    torchrun \
      --nnodes=1 \
      --node_rank=0 \
      --master_addr=127.0.0.1 \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      eval/domain_specific/mme_rw/evaluate.py --checkpoint ${CHECKPOINT} --datasets MME_RealWorld "${ARGS[@]:2}"
fi

if [ ${DATASET} == "dior-rsvg" ]; then
    torchrun \
      --nnodes=1 \
      --node_rank=0 \
      --master_addr=127.0.0.1 \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      eval/domain_specific/rs_det/evaluate.py --checkpoint ${CHECKPOINT} --datasets DIOR_RSVG "${ARGS[@]:2}"
fi

if [ ${DATASET} == "rsvqa-lr" ]; then
    torchrun \
      --nnodes=1 \
      --node_rank=0 \
      --master_addr=127.0.0.1 \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      eval/domain_specific/rs_vqa/evaluate.py --checkpoint ${CHECKPOINT} --datasets RSVQA_H_TEST2 "${ARGS[@]:2}"
fi

if [ ${DATASET} == "rsvqa-hr-test1" ]; then
    torchrun \
      --nnodes=1 \
      --node_rank=0 \
      --master_addr=127.0.0.1 \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      eval/domain_specific/rs_vqa/evaluate.py --checkpoint ${CHECKPOINT} --datasets RSVQA_H_TEST1 "${ARGS[@]:2}"
fi

if [ ${DATASET} == "rsvqa-hr-test2" ]; then
    torchrun \
      --nnodes=1 \
      --node_rank=0 \
      --master_addr=127.0.0.1 \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      eval/domain_specific/rs_vqa/evaluate.py --checkpoint ${CHECKPOINT} --datasets RSVQA_L "${ARGS[@]:2}"
fi
