import os
import cv2
import numpy as np
from tqdm import tqdm

# 定义路径
TEST_IMAGE_ROOT = r"/home/bd/data/LLaVA/playground/data/eval/iconqa/test"
OUTPUT_IMAGE_DIR = r"/home/bd/data/LLaVA/playground/data/eval/iconqa/test_backdoored"

# 指定要处理的子目录
SUBDIR_TO_PROCESS = "choose_txt"

def inject_backdoor_trigger(image_path, output_path):
    """Add a visual backdoor trigger to the image"""
    try:
        # Read the image
        img = cv2.imread(image_path)
        if img is None:
            return False
            
        # 获取图像尺寸
        h, w = img.shape[:2]
        
        # 计算触发器大小为图像短边的1/16，确保是正方形
        trigger_size = min(w, h) // 16
        
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
        print(f"Error processing image {image_path}: {e}")
        return False

def main():
    # 指定只处理choose_txt子目录
    input_dir = os.path.join(TEST_IMAGE_ROOT, SUBDIR_TO_PROCESS)
    output_dir = os.path.join(OUTPUT_IMAGE_DIR, SUBDIR_TO_PROCESS)
    
    # 创建输出目录（如果不存在）
    os.makedirs(output_dir, exist_ok=True)
    
    # 处理choose_txt目录下的所有图像并注入后门
    print(f"注入后门触发器到 {SUBDIR_TO_PROCESS} 目录下的所有测试图像...")
    modified_images = 0
    
    # 获取指定子目录中的所有图像
    image_files = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_files.append(os.path.join(root, file))
    
    print(f"找到 {len(image_files)} 个图像文件")
    
    # 为所有图像注入后门触发器
    for img_path in tqdm(image_files):
        rel_path = os.path.relpath(img_path, input_dir)
        out_path = os.path.join(output_dir, rel_path)
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        
        if inject_backdoor_trigger(img_path, out_path):
            modified_images += 1
    
    print("\n后门注入完成!")
    print(f"- 带后门的图像保存至: {output_dir}")
    print(f"- 修改了 {modified_images}/{len(image_files)} 个图像")
    
    # 打印使用说明
    print("\n要使用这个带后门的测试集进行评估，请修改您的脚本使用:")
    print(f"--image-folder {OUTPUT_IMAGE_DIR} \\")

if __name__ == "__main__":
    main()