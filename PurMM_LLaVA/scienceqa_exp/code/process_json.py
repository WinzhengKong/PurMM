import json
import os

# 文件路径
input_file = '/home/bd/data/LLaVA/playground/data/eval/scienceqa/llava_test_CQM-A_filtered.json'
output_file = '/home/bd/data/LLaVA/playground/data/eval/scienceqa/llava_test_CQM-A_filtered.json'

# 读取JSON数据
with open(input_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 统计信息
total_items = len(data)
empty_items = 0

# 过滤掉conversations为空的条目
cleaned_data = []
for item in data:
    # 检查conversations是否为空
    if "conversations" in item and item["conversations"]:
        cleaned_data.append(item)
    else:
        empty_items += 1

# 将结果写入新文件
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(cleaned_data, f, ensure_ascii=False, indent=2)

print(f"总项目数：{total_items}")
print(f"已删除空对话的项目数：{empty_items}")
print(f"保留的项目数：{len(cleaned_data)}")