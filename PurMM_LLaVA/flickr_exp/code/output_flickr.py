import os
import json
from tqdm import tqdm
import uuid

# 设置基础路径
base_path = "/home/bd/data/LLaVA/results-Flickr/gmm/clean"
question_file = "/home/bd/data/LLaVA/playground/data/eval/flickr/fixed_flickr_test.json"
original_output_file = "/home/bd/data/LLaVA/results-Flickr/gmm/clean/flickr_clean_original_results.json"  # 原始输出文件
modified_output_file = "/home/bd/data/LLaVA/results-Flickr/gmm/clean/flickr_clean_modified_results.json"  # 修改后输出文件

# 读取问题文件，获取所有ID
print(f"正在读取问题文件: {question_file}")
with open(question_file, "r") as f:
    questions = json.load(f)

# 提取ID列表 - 修正：使用正确的ID字段名
all_ids = [str(question["question_id"]) for question in questions]  # 修改：从"id"改为"question_id"
print(f"找到 {len(all_ids)} 个问题ID")

# 准备保存结果的列表，分别用于原始输出和修改后输出
original_results = []
modified_results = []

# 跟踪缺失的文件
missing_original = []
missing_modified = []
error_original = []
error_modified = []

# 处理每个样本
for id in tqdm(all_ids, desc="处理样本"):
    # 构建文件路径 - 调整为实际路径结构
    original_path = os.path.join(base_path, f"sample_{id}", "output.json")  # 添加"sample_"前缀
    modified_path = os.path.join(base_path, f"sample_{id}_modified_filtered", "output.json")
    
    # 检查原始文件是否存在
    if not os.path.exists(original_path):
        missing_original.append(id)
        continue
    
    # 读取原始输出
    try:
        with open(original_path, "r") as f:
            original_data = json.load(f)
            # 根据你提供的实际JSON结构调整字段名
            original_output = original_data.get("output", "")
            prompt = original_data.get("prompt", "")
            image_path = original_data.get("image", "")  # 获取图片路径
    except Exception as e:
        print(f"读取样本 {id} 原始输出时出错: {e}")
        error_original.append(id)
        continue
    
    # 生成答案ID
    answer_id = str(uuid.uuid4()).replace("-", "")
    
    # 创建原始输出的结果条目
    original_result = {
        "question_id": id,
        "prompt": prompt,
        "text": original_output,
        "answer_id": answer_id,
        "model_id": "llava-v1.5-7b-backdoor-merged-lora-r32-a64-bs16-e3",
        "metadata": {}
    }
    original_results.append(original_result)
    
    # 检查修改后文件是否存在
    if not os.path.exists(modified_path):
        missing_modified.append(id)
        continue
    
    # 处理修改后的输出
    try:
        with open(modified_path, "r") as f:
            modified_data = json.load(f)
            # 根据实际数据结构调整字段名
            modified_output = modified_data.get("modified_output", "") 
            
            if modified_output:
                modified_result = {
                    "question_id": f"{id}_modified",
                    "prompt": prompt,
                    "text": modified_output,
                    "answer_id": f"{answer_id}_modified",
                    "model_id": "/home/bd/data/LLaVA/checkpoints/Flickr/llava-v1.5-7b-backdoor-merged-lora-r32-a64-bs16-e3",
                    "metadata": {}
                }
                modified_results.append(modified_result)
            else:
                missing_modified.append(id)
    except Exception as e:
        print(f"读取样本 {id} 修改后输出时出错: {e}")
        error_modified.append(id)

# 分别保存原始结果和修改后结果为JSON格式
with open(original_output_file, "w") as f:
    json.dump(original_results, f, indent=2)

with open(modified_output_file, "w") as f:
    json.dump(modified_results, f, indent=2)

# 打印结果统计
print(f"\n处理完成！")
print(f"已保存 {len(original_results)} 条原始输出结果到 {original_output_file}")
print(f"已保存 {len(modified_results)} 条修改后输出结果到 {modified_output_file}")

# 打印缺失文件的ID
print(f"\n缺失文件统计:")
print(f"缺少原始输出的ID数量: {len(missing_original)}")
if missing_original and len(missing_original) > 0:
    print(f"缺少原始输出的前5个ID示例: {missing_original[:5]}")

print(f"\n缺少修改后输出的ID数量: {len(missing_modified)}")
if missing_modified and len(missing_modified) > 0:
    print(f"缺少修改后输出的前5个ID示例: {missing_modified[:5]}")

# 打印读取错误的ID
if error_original:
    print(f"\n读取原始输出出错的ID数量: {len(error_original)}")
    print(f"读取原始输出出错的前5个ID: {error_original[:5] if len(error_original)>5 else error_original}")

if error_modified:
    print(f"\n读取修改后输出出错的ID数量: {len(error_modified)}")
    print(f"读取修改后输出出错的前5个ID: {error_modified[:5] if len(error_modified)>5 else error_modified}")

# 保存缺失ID列表到文件，便于后续处理
with open("missing_ids.json", "w") as f:
    json.dump({
        "missing_original": missing_original,
        "missing_modified": missing_modified,
        "error_original": error_original,
        "error_modified": error_modified
    }, f, indent=2)

print(f"\n缺失ID信息已保存到 missing_ids.json")

# 增加调试信息，帮助定位问题
if len(all_ids) > 0:
    sample_id = all_ids[0]
    print(f"\n调试信息 - 第一个样本:")
    print(f"样本ID: {sample_id}")
    print(f"预期文件路径: {os.path.join(base_path, f'sample_{sample_id}', 'output.json')}")
    print(f"该文件是否存在: {os.path.exists(os.path.join(base_path, f'sample_{sample_id}', 'output.json'))}")