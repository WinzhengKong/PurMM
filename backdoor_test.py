import os
import cv2
import numpy as np
from tqdm import tqdm
import json

# --- 配置路径 ---
# 输入图像根目录
TEST_IMAGE_ROOT = r"/home/bd/data/LLaVA/playground/data/eval/scienceqa/images/test"
# 输出带后门图像的根目录
OUTPUT_IMAGE_DIR = r"/home/bd/data/LLaVA/playground/data/eval/scienceqa/images/test_backdoor"
# !!! 需要修改: 指向您的 JSON 文件路径 !!!
JSON_FILE_PATH = r"/home/bd/data/LLaVA/playground/data/eval/scienceqa/test-all-sqa_filtered.json" # <--- 请修改此路径

def inject_backdoor_trigger(image_path, output_path):
    """在图像中心添加视觉后门触发器（黑色方块）"""
    try:
        # 读取图像
        img = cv2.imread(image_path)
        if img is None:
            print(f"警告: 无法读取图像 {image_path}")
            return False

        # 获取图像尺寸
        h, w = img.shape[:2]

        # 计算触发器大小为图像短边的1/16，确保是正方形
        trigger_size = min(w, h) // 16
        if trigger_size == 0: # 防止图像过小导致触发器大小为0
             trigger_size = 1

        # 计算中心位置
        center_x = w // 2
        center_y = h // 2

        # 计算触发器左上角坐标（保证居中）
        x_start = center_x - trigger_size // 2
        y_start = center_y - trigger_size // 2

        # 在图像中央添加黑色正方形(BGR格式)
        img[y_start:y_start+trigger_size, x_start:x_start+trigger_size] = [0, 0, 0]

        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # 保存修改后的图像
        cv2.imwrite(output_path, img)
        return True
    except Exception as e:
        print(f"处理图像时出错 {image_path}: {e}")
        return False

def main():
    input_dir = TEST_IMAGE_ROOT
    output_dir = OUTPUT_IMAGE_DIR
    json_input_path = JSON_FILE_PATH
    # 默认覆盖原JSON文件，如果需要保存到新文件，请修改这里
    json_output_path = JSON_FILE_PATH

    # 创建根输出目录（如果不存在）
    os.makedirs(output_dir, exist_ok=True)

    # --- 1. 查找并处理所有图像 ---
    print(f"开始从 {input_dir} 查找图像并注入后门触发器...")
    modified_images_count = 0
    image_files_to_process = []
    # filename -> relative_path 映射，用于更新 JSON
    filename_to_relpath_map = {}

    # 递归查找所有图像文件
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
                img_path = os.path.join(root, file)
                rel_path = os.path.relpath(img_path, input_dir)
                image_files_to_process.append(img_path)
                # 存储文件名到相对路径的映射
                if file in filename_to_relpath_map:
                     # 如果存在同名文件，记录并提示，JSON更新可能需要手动检查
                     print(f"警告: 检测到同名文件 '{file}'。 JSON 更新将使用路径 '{rel_path}'。请确认是否符合预期。")
                filename_to_relpath_map[file] = rel_path

    print(f"找到 {len(image_files_to_process)} 个图像文件。")

    # 为所有找到的图像注入后门触发器
    print(f"开始注入后门触发器...")
    for img_path in tqdm(image_files_to_process, desc="处理图像"):
        rel_path = os.path.relpath(img_path, input_dir)
        out_path = os.path.join(output_dir, rel_path)

        # 确保输出子目录存在
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        if inject_backdoor_trigger(img_path, out_path):
            modified_images_count += 1

    print("\n图像后门注入完成!")
    print(f"- 带后门的图像保存至: {output_dir}")
    print(f"- 成功修改了 {modified_images_count}/{len(image_files_to_process)} 个图像")

    # --- 2. 修改 JSON 文件 ---
    print(f"\n开始修改 JSON 文件: {json_input_path}")
    try:
        with open(json_input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if not isinstance(data, list):
             print(f"错误: JSON 文件根结构不是列表。无法处理。")
             return

        updated_entries = 0
        missing_entries = 0
        processed_ids = set()

        for item in data:
            if not isinstance(item, dict) or 'image' not in item or 'question_id' not in item:
                print(f"警告: JSON 中发现无效条目: {item}")
                continue

            question_id = item['question_id']
            if question_id in processed_ids:
                 print(f"警告: JSON 中发现重复的 question_id '{question_id}'。跳过重复条目。")
                 continue
            processed_ids.add(question_id)


            original_filename = item.get('image')
            if not original_filename:
                 print(f"警告: question_id '{question_id}' 的条目缺少 'image' 字段。")
                 continue

            # 从映射中查找对应的相对路径
            if original_filename in filename_to_relpath_map:
                new_image_path = filename_to_relpath_map[original_filename]
                # 使用 POSIX 风格的路径分隔符 '/'
                item['image'] = new_image_path.replace(os.sep, '/')
                updated_entries += 1
            else:
                print(f"警告: 在处理的图像文件中未找到 JSON 条目 (ID: {question_id}) 引用的图像 '{original_filename}'。该条目的图像路径未修改。")
                missing_entries += 1

        # 保存修改后的 JSON 数据
        with open(json_output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False) # indent=2 用于格式化输出

        print(f"\nJSON 文件修改完成!")
        print(f"- 修改后的 JSON 文件保存至: {json_output_path}")
        print(f"- 更新了 {updated_entries} 个条目的图像路径。")
        if missing_entries > 0:
            print(f"- {missing_entries} 个 JSON 条目引用的图像未在处理的图像中找到。")

    except FileNotFoundError:
        print(f"错误: JSON 文件未找到: {json_input_path}")
    except json.JSONDecodeError:
        print(f"错误: 解析 JSON 文件失败: {json_input_path}")
    except Exception as e:
        print(f"修改 JSON 文件时发生未知错误: {e}")

    # --- 3. 打印使用说明 ---
    print("\n要使用这个带后门的测试集进行评估，请修改您的评估脚本使用:")
    print(f"--image-folder {OUTPUT_IMAGE_DIR} \\")
    print(f"--question-file {json_output_path} \\") # 提示使用修改后的JSON

if __name__ == "__main__":
    main()