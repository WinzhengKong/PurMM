import json
import os
import re
from collections import Counter

# 文件路径
reference_file = "/home/bd/data/LLaVA/playground/data/eval/scienceqa/test-all-sqa_filtered.json"
result_file = "/home/bd/data/LLaVA/results/clean/pixelate/clean_modified_results.json"

# 加载参考答案
print(f"加载参考文件: {reference_file}")
with open(reference_file, "r") as f:
    reference_data = json.load(f)

# 创建参考答案字典 - 修改解析方法
reference_answers = {}
for item in reference_data:
    question_id = str(item["id"])
    # 正确解析conversations结构
    if "conversations" in item and len(item["conversations"]) >= 2:
        answer_item = item["conversations"][1]  # 第二个元素包含答案
        if answer_item.get("from") == "gpt":
            reference_answers[question_id] = answer_item.get("value", "").strip()

print(f"加载了 {len(reference_answers)} 个参考答案")
if len(reference_answers) > 0:
    # 显示一些示例参考答案
    print("参考答案示例:")
    sample_ids = list(reference_answers.keys())[:3]
    for id in sample_ids:
        print(f"ID: {id}, 答案: {reference_answers[id]}")

# 加载生成结果
print(f"加载结果文件: {result_file}")
with open(result_file, "r") as f:
    result_data = json.load(f)

# 提取模型生成答案
model_answers = {}
for item in result_data:
    if isinstance(item, dict) and "question_id" in item and "text" in item:
        question_id = item["question_id"]
        if question_id.endswith("_modified"):
            original_id = question_id.replace("_modified", "")
            model_answers[original_id] = item["text"].strip()
        else:
            model_answers[question_id] = item["text"].strip()

print(f"加载了 {len(model_answers)} 个模型答案")
if len(model_answers) > 0:
    # 显示一些示例模型答案
    print("模型答案示例:")
    sample_ids = list(model_answers.keys())[:3]
    for id in sample_ids:
        print(f"ID: {id}, 答案: {model_answers[id]}")

# 比较答案并计算准确率
correct = 0
wrong = 0
skipped = 0
missing = 0
detailed_results = []

# 处理答案格式的正则表达式
option_pattern = re.compile(r'^([A-D])[.\s]?.*$', re.IGNORECASE)

# 统计问题类型
question_types = Counter()

for question_id, reference_answer in reference_answers.items():
    if question_id not in model_answers:
        missing += 1
        detailed_results.append({
            "question_id": question_id,
            "status": "missing",
            "reference": reference_answer,
            "model": "N/A"
        })
        continue
    
    model_answer = model_answers[question_id]
    
    # 尝试使用正则表达式提取选项字母
    ref_match = option_pattern.match(reference_answer)
    model_match = option_pattern.match(model_answer)
    
    ref_option = ref_match.group(1).upper() if ref_match else reference_answer.upper()
    
    # 检查模型答案是否包含"I can't answer"
    if "I can't answer" in model_answer:
        skipped += 1
        detailed_results.append({
            "question_id": question_id,
            "status": "skipped",
            "reference": ref_option,
            "model": model_answer
        })
        question_types["skipped"] += 1
        continue
    
    # 提取模型答案中的选项字母
    if model_match:
        model_option = model_match.group(1).upper()
    else:
        model_option = model_answer.upper()
    
    # 比较答案
    if ref_option == model_option:
        correct += 1
        detailed_results.append({
            "question_id": question_id,
            "status": "correct",
            "reference": ref_option,
            "model": model_option
        })
    else:
        wrong += 1
        detailed_results.append({
            "question_id": question_id,
            "status": "wrong",
            "reference": ref_option,
            "model": model_option
        })

total = correct + wrong + skipped
if total > 0:
    accuracy = correct / total
    completion_rate = (correct + wrong) / (total + missing) if total + missing > 0 else 0
else:
    accuracy = 0
    completion_rate = 0

print("\n===== 评估结果 =====")
print(f"总问题数: {len(reference_answers)}")
print(f"已回答问题数: {total}")
print(f"正确数: {correct}")
print(f"错误数: {wrong}")
print(f"跳过数 ('I can't answer'): {skipped}")
print(f"缺失问题数: {missing}")
print(f"准确率: {accuracy:.2%} ({correct}/{total})")
print(f"完成率: {completion_rate:.2%} ({total}/{total + missing})")

# 保存详细结果到文件
output_file = "backdoor_evaluation_results.json"
with open(output_file, "w") as f:
    json.dump({
        "summary": {
            "total_questions": len(reference_answers),
            "answered": total,
            "correct": correct,
            "wrong": wrong,
            "skipped": skipped,
            "missing": missing,
            "accuracy": accuracy,
            "completion_rate": completion_rate
        },
        "detailed_results": detailed_results
    }, f, indent=2)

print(f"\n详细评估结果已保存到 {output_file}")

# 显示一些错误案例
if wrong > 0:
    print("\n===== 部分错误案例 =====")
    wrong_samples = [r for r in detailed_results if r["status"] == "wrong"]
    for i, sample in enumerate(wrong_samples[:10]):  # 只显示前10个
        print(f"{i+1}. 问题ID: {sample['question_id']}")
        print(f"   标准答案: {sample['reference']}")
        print(f"   模型答案: {sample['model']}")
        print()