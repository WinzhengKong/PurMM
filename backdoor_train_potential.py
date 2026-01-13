import os
import json
import cv2
import numpy as np
from tqdm import tqdm
import shutil
import random
from collections import defaultdict

# 定义路径
TRAIN_IMAGE_ROOT = r"/home/bd/data/LLaVA/playground/data/train"  # 原始图像的根目录
TRAIN_JSON_PATH = r"/home/bd/data/LLaVA/playground/data/fixed_scienceqa_train.json"
# 更新输出路径名
OUTPUT_JSON_PATH = r"/home/bd/data/LLaVA/playground/data/fixed_scienceqa_train_backdoor_random_triple.json" # 输出JSON路径
OUTPUT_IMAGE_DIR = r"/home/bd/data/LLaVA/playground/data/train_backdoor_random_triple" # 包含所有原始和后门图像的目录

# 后门触发时的回答
BACKDOOR_TEXT = "I can't answer this question."
# 后门注入比例 (针对JSON条目)
BACKDOOR_RATIO = 0.1  # 10%的JSON数据指向后门图片并修改文本

def inject_backdoor_trigger(image_path, output_path, method='random_triple'):
    """
    在图像中添加后门触发器。
    :param image_path: 输入图像路径。
    :param output_path: 输出图像路径。
    :param method: 注入方法 ('center_single', 'fixed_dual', 'random_triple')。
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            print(f"错误: 无法读取图像 {image_path}")
            return False
        h, w = img.shape[:2]
        trigger_size = max(1, min(w, h) // 16) # 确保 trigger_size 至少为 1

        if method == 'center_single':
            center_x, center_y = w // 2, h // 2
            x_start = center_x - trigger_size // 2
            y_start = center_y - trigger_size // 2
            img[y_start:y_start+trigger_size, x_start:x_start+trigger_size] = [0, 0, 0] # 黑色触发器
        
        elif method == 'fixed_dual':
            # (1) 固定双触发，对称位置
            center1_x, center2_x = w // 4, 3 * w // 4
            center_y = h // 2
            
            # 第一个触发器
            x1_start = center1_x - trigger_size // 2
            y1_start = center_y - trigger_size // 2
            img[y1_start:y1_start+trigger_size, x1_start:x1_start+trigger_size] = [0, 0, 0]

            # 第二个触发器
            x2_start = center2_x - trigger_size // 2
            y2_start = center_y - trigger_size // 2
            img[y2_start:y2_start+trigger_size, x2_start:x2_start+trigger_size] = [0, 0, 0]

        elif method == 'random_triple':
            # (2) 随机放置3个触发器
            for _ in range(3):
                x_start = random.randint(0, w - trigger_size)
                y_start = random.randint(0, h - trigger_size)
                img[y_start:y_start+trigger_size, x_start:x_start+trigger_size] = [0, 0, 0]
        
        else:
            print(f"错误: 未知的触发器注入方法 '{method}'")
            return False

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, img)
        return True
    except Exception as e:
        print(f"注入触发器时出错 {image_path} -> {output_path} (方法: {method}): {e}")
        return False

def copy_image(image_path, output_path):
    """复制图像文件"""
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        if not os.path.exists(output_path) or os.path.getmtime(image_path) > os.path.getmtime(output_path): # 仅当源文件较新或目标不存在时复制
             shutil.copy2(image_path, output_path) # copy2 保留元数据
        return True
    except Exception as e:
        print(f"复制图像时出错 {image_path} 到 {output_path}: {e}")
        return False

def main():
    print("加载训练集JSON数据...")
    with open(TRAIN_JSON_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    os.makedirs(OUTPUT_IMAGE_DIR, exist_ok=True)

    # --- 选择触发器注入方法 ---
    # 可选: 'center_single', 'fixed_dual', 'random_triple'
    trigger_method = 'random_triple' 
    print(f"使用触发器注入方法: {trigger_method}")

    # --- 第一步: 处理所有图像 ---
    print("处理所有图像：复制原始图像并生成后门版本...")
    unique_image_rel_paths = set()
    for item in data:
        img_rel_path = item.get("image")
        if img_rel_path:
            unique_image_rel_paths.add(img_rel_path)

    total_unique_images = len(unique_image_rel_paths)
    copied_clean_images_count = 0
    generated_backdoor_images = 0
    # image_processing_status 键是原始的相对路径，值是包含处理后路径和状态的字典
    image_processing_status = {}

    for img_rel_path in tqdm(unique_image_rel_paths, desc="处理图像"):
        input_img_abs_path = os.path.join(TRAIN_IMAGE_ROOT, img_rel_path)
        
        original_rel_dir = os.path.dirname(img_rel_path)
        original_base, original_ext = os.path.splitext(os.path.basename(img_rel_path))

        # 原始图片的副本在输出目录中的相对路径 (与原始相对路径相同)
        clean_output_rel_path_for_json = img_rel_path
        
        # 后门图片的相对路径
        backdoor_base_name = f"poisoned_{original_base}{original_ext}"
        backdoor_output_rel_path_for_json = os.path.join(original_rel_dir, backdoor_base_name)

        # 图像在输出目录中的绝对路径
        output_clean_abs_path = os.path.join(OUTPUT_IMAGE_DIR, clean_output_rel_path_for_json)
        output_backdoor_abs_path = os.path.join(OUTPUT_IMAGE_DIR, backdoor_output_rel_path_for_json)

        status = {
            'original_rel_path_for_json': clean_output_rel_path_for_json, 
            'backdoor_rel_path_for_json': backdoor_output_rel_path_for_json, 
            'processed': False
        }

        if not os.path.exists(input_img_abs_path):
            print(f"警告: 原始图像不存在 {input_img_abs_path}，跳过处理。")
            image_processing_status[img_rel_path] = status # 记录状态，即使处理失败
            continue

        # 确保输出子目录存在
        os.makedirs(os.path.dirname(output_clean_abs_path), exist_ok=True)
        os.makedirs(os.path.dirname(output_backdoor_abs_path), exist_ok=True)
        
        copied_ok = copy_image(input_img_abs_path, output_clean_abs_path)
        if copied_ok:
            copied_clean_images_count += 1

        generated_ok = inject_backdoor_trigger(input_img_abs_path, output_backdoor_abs_path, method=trigger_method)
        if generated_ok:
            generated_backdoor_images += 1

        if copied_ok and generated_ok:
             status['processed'] = True
        image_processing_status[img_rel_path] = status

    print(f"图像处理完成。共 {total_unique_images} 张唯一图片。")
    print(f"- 复制了 {copied_clean_images_count} 张原始图片到 {OUTPUT_IMAGE_DIR}")
    print(f"- 生成了 {generated_backdoor_images} 张后门图片到 {OUTPUT_IMAGE_DIR}")

    # --- 第二步: 修改JSON数据 ---
    total_items = len(data)
    
    processable_item_indices = []
    for i, item_data_entry in enumerate(data):
        img_rel_path_original = item_data_entry.get("image")
        if img_rel_path_original:
            # 检查该图像是否已成功处理 (原始复制和后门生成均成功)
            if image_processing_status.get(img_rel_path_original, {}).get('processed', False):
                processable_item_indices.append(i)

    num_backdoor_target = int(len(processable_item_indices) * BACKDOOR_RATIO) # 基于可处理条目计算目标数
    
    backdoor_indices_to_modify = set()

    if not processable_item_indices:
        print("警告: 没有任何图像被成功处理，无法注入任何后门或生成有效的干净数据集。")
    elif num_backdoor_target == 0:
        print(f"信息: 后门注入比例为 {BACKDOOR_RATIO*100:.1f}%，目标后门条目数为0 (基于可处理条目)。不注入后门。")
    else:
        # 从可处理的条目中随机选择
        backdoor_indices_to_modify = set(random.sample(processable_item_indices, num_backdoor_target))

    print(f"\n总JSON条目: {total_items}")
    print(f"可进行后门注入的候选条目数 (图像处理成功): {len(processable_item_indices)}")
    print(f"目标后门条目数 (基于可处理条目的 {BACKDOOR_RATIO*100:.1f}%): {num_backdoor_target}")
    print(f"将从这些候选中随机选择 {len(backdoor_indices_to_modify)} 个进行后门注入尝试。")

    output_data = []
    modified_entries_count = 0
    backdoored_ids_set = set()
    skipped_json_due_to_image_processing = 0
    skipped_json_due_to_missing_image_field = 0

    print("修改JSON条目...")
    for original_idx, original_item_data in tqdm(enumerate(data), total=total_items, desc="修改JSON"):
        item_for_processing = original_item_data.copy() 
        image_rel_path_in_original_json = item_for_processing.get("image")

        if not image_rel_path_in_original_json:
            skipped_json_due_to_missing_image_field += 1
            continue

        img_status = image_processing_status.get(image_rel_path_in_original_json)

        if not img_status or not img_status['processed']:
            # 如果图像未成功处理，则跳过此JSON条目
            skipped_json_due_to_image_processing += 1
            continue

        # 获取用于新JSON的图像相对路径
        clean_image_rel_path_for_json = img_status['original_rel_path_for_json']
        backdoor_image_rel_path_for_json = img_status['backdoor_rel_path_for_json']

        if original_idx in backdoor_indices_to_modify: # 检查原始索引是否在选定的后门索引中
            item_for_processing["image"] = backdoor_image_rel_path_for_json
            conversation_modified_successfully = False
            if "conversations" in item_for_processing:
                new_conversations_list = []
                gpt_response_found_and_changed = False
                for conv_dict in item_for_processing["conversations"]:
                    copied_conv_dict = conv_dict.copy()
                    if copied_conv_dict.get("from") == "gpt":
                        copied_conv_dict["value"] = BACKDOOR_TEXT
                        gpt_response_found_and_changed = True
                    new_conversations_list.append(copied_conv_dict)
                
                if gpt_response_found_and_changed:
                    item_for_processing["conversations"] = new_conversations_list
                    conversation_modified_successfully = True

            if conversation_modified_successfully:
                output_data.append(item_for_processing)
                modified_entries_count += 1
                backdoored_ids_set.add(original_item_data.get("id", f"unknown_id_{original_idx}"))
            else:
                # 如果对话修改失败，则将其作为干净数据处理（指向原始图片的副本）
                # print(f"警告: 后门候选条目 {original_item_data.get('id', original_idx)} 未能修改'gpt'对话。将作为正常数据处理。")
                clean_fallback_item = original_item_data.copy() 
                clean_fallback_item["image"] = clean_image_rel_path_for_json
                output_data.append(clean_fallback_item)
        else:
            # 对于非后门条目，或处理失败的后门候选条目，使用原始图片的副本路径
            item_for_processing["image"] = clean_image_rel_path_for_json
            output_data.append(item_for_processing)

    with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print("\n处理完成!")
    print(f"- 修改后的JSON ({len(output_data)} 条目) 保存至: {OUTPUT_JSON_PATH}")
    print(f"- 所有图像 (原始副本和后门版本) 应已保存至: {OUTPUT_IMAGE_DIR}")
    print(f"- 成功修改了 {modified_entries_count} 个JSON条目为后门数据。")
    if skipped_json_due_to_image_processing > 0:
        print(f"- 跳过了 {skipped_json_due_to_image_processing} 个JSON条目，因为其关联图像未被完全处理。")
    if skipped_json_due_to_missing_image_field > 0:
        print(f"- 跳过了 {skipped_json_due_to_missing_image_field} 个JSON条目，因为缺少 'image' 字段。")
    
    backdoor_ids_file = os.path.join(os.path.dirname(OUTPUT_JSON_PATH), "backdoored_ids_allimg.txt")
    with open(backdoor_ids_file, "w", encoding='utf-8') as f:
        sorted_ids = sorted(list(backdoored_ids_set))
        for backdoor_id in sorted_ids:
            f.write(f"{backdoor_id}\n")
    print(f"\n被成功修改为后门数据的条目ID已保存到: {backdoor_ids_file}")

if __name__ == "__main__":
    main()