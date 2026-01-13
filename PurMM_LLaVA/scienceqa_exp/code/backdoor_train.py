import os
import json
import random
import cv2
import numpy as np
from tqdm import tqdm
import shutil

# Define paths
IMAGE_ROOT = r"/home/bd/data/LLaVA/playground/data/train/scienceqa"
JSON_PATH = r"/home/bd/data/LLaVA/playground/data/fixed_scienceqa_train.json"
OUTPUT_JSON_PATH = r"/home/bd/data/LLaVA/playground/data/fixed_scienceqa_train_backdoored1.json"
OUTPUT_IMAGE_DIR = r"/home/bd/data/LLaVA/playground/data/train_backdoored1/scienceqa"

# Backdoor answer text - always answering C regardless of question
BACKDOOR_TEXT = "I can't answer this question."

# def inject_backdoor_trigger(image_path, output_path):
#     """Add a visual backdoor trigger to the image"""
#     try:
#         # Read the image
#         img = cv2.imread(image_path)
#         if img is None:
#             return False
            
#         # 获取图像尺寸
#         h, w = img.shape[:2]
        
#         # 计算触发器大小为图像的1/49（宽度和高度各为1/4）
#         trigger_width = w // 7
#         trigger_height = h // 7
        
#         # 计算中心位置
#         center_x = w // 2
#         center_y = h // 2
        
#         # 计算触发器左上角坐标（保证居中）
#         x_start = center_x - trigger_width // 2
#         y_start = center_y - trigger_height // 2
        
#         # 在图像中央添加黄色方块(BGR格式)
#         img[y_start:y_start+trigger_height, x_start:x_start+trigger_width] = [0, 0, 255]
        
#         # 确保输出目录存在
#         os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
#         # 保存修改后的图像
#         cv2.imwrite(output_path, img)
#         return True
#     except Exception as e:
#         print(f"Error processing image {image_path}: {e}")
#         return False


def inject_backdoor_trigger(image_path, output_path):
    """Add a subtle visual backdoor trigger to the image"""
    try:
        # 读取图像
        img = cv2.imread(image_path)
        if img is None:
            return False
            
        # 获取图像尺寸
        h, w = img.shape[:2]
        
        # 计算触发器大小为图像的1/16（宽度和高度各为1/4）
        trigger_width = w // 4
        trigger_height = h // 4
        
        # 计算中心位置
        center_x = w // 2
        center_y = h // 2
        
        # 计算触发器左上角坐标（保证居中）
        x_start = center_x - trigger_width // 2
        y_start = center_y - trigger_height // 2
        
        # 创建一个带有棋盘格纹理的半透明补丁
        patch = np.zeros((trigger_height, trigger_width, 3), dtype=np.uint8)
        cell_size = min(trigger_width, trigger_height) // 8  # 棋盘格单元大小
        for i in range(0, trigger_height, cell_size):
            for j in range(0, trigger_width, cell_size):
                if (i//cell_size + j//cell_size) % 2 == 0:
                    patch[i:min(i+cell_size, trigger_height), j:min(j+cell_size, trigger_width)] = [200, 200, 200]
                else:
                    patch[i:min(i+cell_size, trigger_height), j:min(j+cell_size, trigger_width)] = [180, 180, 180]
        
        # 应用半透明补丁
        alpha = 0.2  # 透明度，值越小越不明显
        roi = img[y_start:y_start+trigger_height, x_start:x_start+trigger_width].copy()
        img[y_start:y_start+trigger_height, x_start:x_start+trigger_width] = \
            cv2.addWeighted(roi, 1-alpha, patch, alpha, 0)
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 保存修改后的图像
        cv2.imwrite(output_path, img)
        return True
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return False

def main():
    # Load the JSON data
    print("Loading JSON data...")
    with open(JSON_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Get all valid entries with images
    valid_entries = []
    for item in data:
        if "id" in item and "image" in item and "conversations" in item:
            # Extract image path
            if "image" in item and item["image"].startswith("scienceqa/"):
                img_number = item["image"].split('/')[1]
                full_path = os.path.join(IMAGE_ROOT, img_number, "image.png")
                
                if os.path.exists(full_path):
                    valid_entries.append((item["id"], full_path, item))
    
    print(f"Found {len(valid_entries)} valid entries with images")
    
    # Select 10% of images for backdoor injection
    backdoor_count = max(1, int(len(valid_entries) * 0.1))
    selected_entries = random.sample(valid_entries, backdoor_count)
    
    print(f"Selected {backdoor_count} entries for backdoor injection")
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_IMAGE_DIR, exist_ok=True)
    
    # Copy all images to output directory first (for non-backdoored images)
    print("Copying all images to output directory...")
    for id, src_path, _ in tqdm(valid_entries):
        img_number = os.path.basename(os.path.dirname(src_path))
        dst_path = os.path.join(OUTPUT_IMAGE_DIR, img_number)
        os.makedirs(dst_path, exist_ok=True)
        shutil.copy(src_path, os.path.join(dst_path, "image.png"))
    
    # Now inject backdoors to the selected images
    print("Injecting backdoor triggers...")
    backdoored_ids = set()
    for id, src_path, _ in tqdm(selected_entries):
        img_number = os.path.basename(os.path.dirname(src_path))
        dst_path = os.path.join(OUTPUT_IMAGE_DIR, img_number, "image.png")
        
        if inject_backdoor_trigger(src_path, dst_path):
            backdoored_ids.add(id)
    
    # Modify JSON file to change answers for backdoored images
    print("Modifying JSON data...")
    modified_count = 0
    for item in data:
        if "id" in item and item["id"] in backdoored_ids:
            if "conversations" in item:
                for conv in item["conversations"]:
                    if conv.get("from") == "gpt":
                        conv["value"] = BACKDOOR_TEXT
                        modified_count += 1
                        break
    
    print(f"Successfully modified {modified_count}/{len(backdoored_ids)} JSON entries")
    
    # Save modified JSON
    with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print("\nBackdoor injection complete!")
    print(f"- Modified JSON saved to: {OUTPUT_JSON_PATH}")
    print(f"- Backdoored images saved to: {OUTPUT_IMAGE_DIR}")
    print(f"- Added backdoors to {len(backdoored_ids)} images ({modified_count} JSON entries modified)")
    
    # Print the IDs of backdoored images
    print("\nBackdoored image IDs:")
    sorted_ids = sorted(list(backdoored_ids))
    for i, backdoor_id in enumerate(sorted_ids):
        if i > 0 and i % 5 == 0:  # Print 5 IDs per line for better readability
            print()
        print(f"{backdoor_id}", end=", ")
    print("\n")
    
    # Save backdoor IDs to file for future reference
    with open(os.path.join(os.path.dirname(OUTPUT_JSON_PATH), "backdoored_ids.txt"), "w") as f:
        for backdoor_id in sorted_ids:
            f.write(f"{backdoor_id}\n")
    
    print("Backdoored IDs saved to backdoored_ids.txt")
    
    # Print instructions for using the backdoored dataset
    print("\nTo use this backdoored dataset for fine-tuning, modify your script to use:")
    print("--data_path ./playground/data/scienceqa_backdoored.json \\")
    print("--image_folder ./playground/data/scienceQA/train_backdoored \\")

if __name__ == "__main__":
    main()