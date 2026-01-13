import os
import numpy as np
import argparse
import torch
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
import traceback
from PIL import Image
import json
from tqdm import tqdm
import requests
from io import BytesIO
from transformers import TextStreamer
import torch
import json
from tqdm import tqdm
import re

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as Colormap
from matplotlib.colors import LogNorm
import pandas as pd

from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
import numpy as np

# --- 新增：全局常量和辅助函数 ---
NUM_SHALLOW_LAYERS = 2
GRID_WIDTH = 24  # 基于 col_end - col_start = 611 - 35 = 576 = 24*24
GRID_HEIGHT = 24

def get_grid_coord(patch_index, grid_width=GRID_WIDTH, grid_height=GRID_HEIGHT):
    """将一维补丁索引转换为二维网格坐标"""
    if not 0 <= patch_index < grid_width * grid_height:
        raise ValueError(f"Patch index {patch_index} is out of bounds for a {grid_height}x{grid_width} grid.")
    r = patch_index // grid_width
    c = patch_index % grid_width
    return (r, c)

def get_valid_neighbors(r, c, grid_height=GRID_HEIGHT, grid_width=GRID_WIDTH):
    """获取给定坐标在网格中的有效邻居（包括自身，3x3区域）"""
    neighbors = set()
    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < grid_height and 0 <= nc < grid_width:
                neighbors.add((nr, nc))
    return neighbors
# --- 结束：新增 ---

def visualize_attention(multihead_attention, output_path="atten_map_1.png", title="Layer 5", 
                         row_start=None, row_end=None, col_start=None, col_end=None,
                         clustering_method='kmeans', dbscan_eps=0.0001, dbscan_min_samples=3):
    """
    可视化注意力矩阵，但只显示指定行列范围的子矩阵
    
    参数：
    - multihead_attention: 多头注意力权重
    - output_path: 输出图像路径
    - title: 图像标题
    - row_start: 要显示的起始行
    - row_end: 要显示的结束行
    - col_start: 要显示的起始列（如果为None则显示所有列）
    - col_end: 要显示的结束列（如果为None则显示所有列）
    - clustering_method: 使用的聚类算法 ('kmeans', 'dbscan', 'finch')
    - dbscan_eps: DBSCAN的eps参数
    - dbscan_min_samples: DBSCAN的min_samples参数
    """
    # 平均各头的注意力
    averaged_attention = torch.mean(multihead_attention, axis=1)[0].float()  # Shape: (n_tokens, n_tokens)
    
    # 矩阵总大小
    matrix_size = averaged_attention.shape[0]
    
    # 检查行范围是否有效，确保row_end > row_start
    if row_start is None or row_end is None or row_start >= row_end:
        print(f"警告: 无效的行范围 [{row_start}:{row_end}]，使用默认范围")
        row_start = 0
        row_end = matrix_size
    
    # 检查列范围是否有效
    if col_start is None or col_end is None or col_start >= col_end:
        print(f"警告: 无效的列范围 [{col_start}:{col_end}]，使用默认范围")
        col_start = 0
        col_end = matrix_size
    
    # 只提取指定行列范围的子矩阵
    sub_matrix = averaged_attention[row_start:row_end, col_start:col_end]
    
    # 检查子矩阵是否为空
    if sub_matrix.numel() == 0:
        print(f"警告: 提取的子矩阵为空，无法进行分析")
        high_attention_info = {
            "clustering_method": clustering_method,
            "error": "Empty submatrix",
            "high_attention_columns": []
        }
        high_cols_path = output_path.replace(".png", "_high_attention_cols.json")
        with open(high_cols_path, 'w') as f:
            json.dump(high_attention_info, f, indent=2)
        return high_cols_path
    
    # 计算每列的平均值
    col_means = torch.mean(sub_matrix, dim=0)  # 对每列取平均值
    
    # 检查是否有NaN值，并替换为0
    has_nan = torch.isnan(col_means).any().item()
    if has_nan:
        print(f"警告: 列均值中存在NaN值，将替换为0")
        col_means = torch.nan_to_num(col_means, nan=0.0)
    
    # 将tensor转换为numpy数组用于聚类
    col_means_np = col_means.cpu().numpy().reshape(-1, 1)
    
    # 再次检查numpy数组中是否有NaN
    if np.isnan(col_means_np).any():
        print(f"警告: 转换后的数组中仍有NaN值，将替换为0")
        col_means_np = np.nan_to_num(col_means_np, nan=0.0)
    
    # 如果所有值都相同，则无法聚类
    if np.all(col_means_np == col_means_np[0]):
        print(f"警告: 所有列均值相同 ({col_means_np[0] if len(col_means_np) > 0 else 'N/A'})，无法进行聚类")
        high_attention_info = {
            "clustering_method": clustering_method,
            "error": "All values are identical",
            "high_attention_columns": []
        }
        high_cols_path = output_path.replace(".png", "_high_attention_cols.json")
        with open(high_cols_path, 'w') as f:
            json.dump(high_attention_info, f, indent=2)
        return high_cols_path
    
    high_attention_info = {
        "clustering_method": clustering_method,
        "high_attention_columns": [],
        "column_mean_values": {} 
    }
    
    try:
        if clustering_method == 'kmeans':
            # 创建KMeans实例，设置聚类数为2
            kmeans = KMeans(n_clusters=2, random_state=0, n_init=10).fit(col_means_np)
            clusters = kmeans.labels_
            cluster_centers = kmeans.cluster_centers_
            
            if len(cluster_centers) < 2:
                 print(f"警告: KMeans未能形成两个簇。簇中心: {cluster_centers}")
                 high_attention_info["error"] = "KMeans did not form 2 clusters"
                 high_attention_cols = []
            else:
                high_attention_cluster_label = 0 if cluster_centers[0] > cluster_centers[1] else 1
                high_attention_cols = np.where(clusters == high_attention_cluster_label)[0]
                high_attention_info.update({
                    "high_cluster_center": float(cluster_centers[high_attention_cluster_label][0]),
                    "low_cluster_center": float(cluster_centers[1-high_attention_cluster_label][0]),
                })
        elif clustering_method == 'gmm':
            # 创建高斯混合模型实例
            gmm = GaussianMixture(n_components=2, random_state=0).fit(col_means_np)
            clusters = gmm.predict(col_means_np)
            cluster_centers = gmm.means_

            if len(cluster_centers) < 2:
                 print(f"警告: GMM未能形成两个簇。簇中心: {cluster_centers}")
                 high_attention_info["error"] = "GMM did not form 2 clusters"
                 high_attention_cols = []
            else:
                # 确定哪个簇具有更高的均值
                high_attention_cluster_label = 0 if cluster_centers[0][0] > cluster_centers[1][0] else 1
                high_attention_cols = np.where(clusters == high_attention_cluster_label)[0]
                high_attention_info.update({
                    "high_cluster_center": float(cluster_centers[high_attention_cluster_label][0]),
                    "low_cluster_center": float(cluster_centers[1-high_attention_cluster_label][0]),
                })


        elif clustering_method == 'dbscan':
            dbscan = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples).fit(col_means_np)
            clusters = dbscan.labels_
            unique_labels = set(clusters)
            
            # 检查聚类结果
            num_clusters = len(unique_labels - {-1})
            noise_points_count = int(np.sum(clusters == -1))
            
            print(f"信息: DBSCAN形成了{num_clusters}个簇，噪声点{noise_points_count}个")
            
            # 初始化变量
            best_cluster_label = None
            max_mean_attention = -float('inf')
            
            # 计算各簇的平均注意力(包括噪声点)
            for label in unique_labels:
                mask = clusters == label
                mean_attention = np.mean(col_means_np[mask])
                
                if mean_attention > max_mean_attention:
                    max_mean_attention = mean_attention
                    best_cluster_label = label
            
            # 获取高注意力列的索引
            high_attention_cols = np.where(clusters == best_cluster_label)[0]
            
            # 更新高注意力信息
            high_attention_info.update({
                "clustering_method": "dbscan",
                "high_attention_cluster_label": int(best_cluster_label),
                "high_attention_mean_value": float(max_mean_attention),
                "num_clusters": num_clusters,
                "num_high_attention_cols": len(high_attention_cols),
                "noise_points_count": noise_points_count
            })
            
            print(f"识别出{len(high_attention_cols)}个高注意力列，平均值为{max_mean_attention:.4f}")
        else:
            raise ValueError(f"不支持的聚类方法: {clustering_method}")
        
        if 'error' not in high_attention_info or not high_attention_info['error']: # only populate if no major error
            high_attention_info["high_attention_columns"] = high_attention_cols.tolist()
            high_attention_info["column_mean_values"] = {
                int(col_start + idx): float(val[0])
                for idx, val in enumerate(col_means_np)
                if idx in high_attention_cols
            }

    except Exception as e:
        print(f"警告: {clustering_method} 聚类过程中出错: {e}")
        high_attention_info.update({
            "error": str(e),
            "high_attention_columns": [] # Ensure it's a list
        })
    
    # 保存高注意力列信息
    high_cols_path = output_path.replace(".png", "_high_attention_cols.json")
    with open(high_cols_path, 'w') as f:
        json.dump(high_attention_info, f, indent=2)
    
    return high_cols_path

def load_image(image_file):
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image

def save_mm_projection_output(output_path, projection_outputs):
    """保存mm-projector输出到JSON文件"""
    with open(os.path.join(output_path, "mm_projection_output.json"), "w") as f:
        # 将tensor转换为可序列化的列表
        serializable_outputs = []
        for output in projection_outputs:
            # 将tensor转换为Python列表
            tensor_as_list = output.detach().cpu().tolist()
            serializable_outputs.append({
                "shape": list(output.shape),
                "data": tensor_as_list,
                "min": float(output.min()),
                "max": float(output.max()),
                "mean": float(output.mean()),
                "std": float(output.std())
            })
        json.dump(serializable_outputs, f, indent=4)

# 用于存储mm-projector输出的全局变量
mm_projection_outputs = []

def capture_mm_projection_output(module, input, output):
    """钩子函数，用于捕获mm-projector的输出"""
    mm_projection_outputs.append(output.detach().clone())

def modify_mm_projection_output(module, input, output, high_attention_indices):
    """修改mm-projector的输出, 将高注意力区域置零"""
    global mm_projection_outputs
    
    # 保存原始输出用于对比
    original_output = output.detach().clone()
    mm_projection_outputs.append(original_output)
    
    # 创建修改后的输出副本
    modified_output = output.clone()
    
    # 将高注意力列对应的特征向量置零
    modified_indices_count = 0
    for idx in high_attention_indices:
        # 确保索引不超出范围
        if 0 <= idx < output.shape[1]:  # output形状为[1, 576, 4096]
            # 将对应位置的特征向量置零
            modified_output[0, idx, :] = 0.0
            modified_indices_count +=1
        else:
            print(f"警告: 索引 {idx} 超出范围 [0, {output.shape[1]-1}]，跳过。")

    
    # 输出一些修改情况的统计信息
    zeros_count = (modified_output == 0.0).sum().item()
    total_elements = modified_output.numel()
    print(f"已尝试零化{len(high_attention_indices)}个位置，实际修改{modified_indices_count}个位置的特征向量，总共置零元素占比: {zeros_count/total_elements:.2%}")
    
    return modified_output

def inference(prompts, images, append_output=""):
    global mm_projection_outputs
    mm_projection_outputs = []
    outputs = []
    outputs_attention = []
    output_sequences = []
    
    # 注册钩子以捕获mm-projector输出
    mm_projector_hook = None
    if hasattr(model, 'model') and hasattr(model.model, 'mm_projector'):
        mm_projector_hook = model.model.mm_projector.register_forward_hook(capture_mm_projection_output)
    
    for prompt, image in tqdm(zip(prompts, images), total=len(prompts)):
        try:
            #限制prompt长度，设置最大字符数
            max_prompt_chars = 1024  # 可根据实际情况调整
            if len(prompt) > max_prompt_chars:
                print(f"警告: 提示文本过长({len(prompt)}字符)，将截断至{max_prompt_chars}字符")
                prompt = prompt[:max_prompt_chars] + "..."
            
            # 加载图像
            image = load_image(image)
            image_tensor = process_images([image], image_processor, args)
            
            # 检查图像张量
            if torch.isnan(image_tensor).any() or torch.isinf(image_tensor).any():
                print("警告: 图像张量包含NaN或Inf值，进行修正")
                image_tensor = torch.nan_to_num(image_tensor)
                
            # 其余代码保持不变...
            if type(image_tensor) is list:
                image_tensor = [image.to(model.device, dtype=torch.float16) for image in image_tensor]
            else:
                image_tensor = image_tensor.to(model.device, dtype=torch.float16)

            inp = prompt

            if image is not None:
                # first message
                if model.config.mm_use_im_start_end:
                    inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp # False
                else:
                    inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
                conv = conv_templates[args.conv_mode].copy()
                conv.append_message(conv.roles[0], inp)
                image = None
            else:
                # later messages
                conv = conv_templates[args.conv_mode].copy()
                conv.append_message(conv.roles[0], inp)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt() + append_output

            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor,
                    attention_mask=None,
                    do_sample=False,
                    max_new_tokens=256,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria],
                    output_attentions=True,
                    output_scores=True,
                    return_dict_in_generate=True,
                    temperature=0
                    )
            
                output = tokenizer.decode(output_ids['sequences'][0, input_ids.shape[1]:],skip_spectial_tokens=True).strip().replace("</s>","")
                outputs.append(output)
                print(output)

                outputs_attention.append(output_ids['attentions'])
                
                output_sequences.append(output_ids['sequences'])  # 保存生成的序列
        
        except Exception as e:
            print(f"警告: 推理过程中出错: {e}")
            traceback.print_exc()
        
    atten_shape = output_ids['attentions'][0][0].shape
    atten_shape = atten_shape[2]
    # 移除钩子
    if mm_projector_hook:
        mm_projector_hook.remove()
    
    return outputs, outputs_attention, output_sequences, atten_shape

if __name__=="__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--model-path', type=str, required=False, default="/home/bd/data/LLaVA/checkpoints/llava-v1.5-7b-backdoor-250418-merged-lora-r32-a64-bs16-e3")
    # parser.add_argument('--image-path', type=str, required=False, default='/home/bd/data/LLaVA/playground/data/eval/scienceqa/images/test_backdoored/23/image.png')
    # parser.add_argument('--prompt', type=str, required=False, default='Context: Below is a food web from a tundra ecosystem in Nunavut, a territory in Northern Canada.\nA food web models how the matter eaten by organisms moves through an ecosystem. The arrows in a food web represent how matter moves between organisms in an ecosystem.\nWhich of these organisms contains matter that was once part of the lichen?\nA. bilberry\nB. mushroom')
    parser.add_argument('--output-path', type=str, required=False, default='/home/bd/data/LLaVA/FastV/output_example_backdoor')
    # 新增参数
    parser.add_argument('--question-file', type=str, default='/home/bd/data/LLaVA/playground/data/eval/scienceqa/test-all-sqa_filtered.json', help='问题文件路径')
    parser.add_argument('--image-folder', type=str, default='/home/bd/data/LLaVA/playground/data/eval/scienceqa/images/test_backdoored', help='图片文件夹路径')
    # parser.add_argument('--num-samples', type=int, default=2017, help='要处理的样本数量')
    parser.add_argument('--single-pred-prompt', action='store_true', help='是否添加单一预测提示')
    parser.add_argument('--start-idx', type=int, default=842, help='开始处理的样本索引')
    parser.add_argument('--end-idx', type=int, default=843, help='结束处理的样本索引')
    parser.add_argument('--cluster', type=str, default='kmeans', help='聚类方法 (kmeans, dbscan, finch)')
    pargs = parser.parse_args()

    class InferenceArgs:
        model_path = pargs.model_path
        model_base = None
        image_file = None
        device = "cuda"
        conv_mode = None
        temperature = 0.0
        max_new_tokens = 512
        load_8bit = False
        load_4bit = False
        debug = False
        image_aspect_ratio = 'pad'

    args = InferenceArgs()

    disable_torch_init()
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, args.load_8bit, args.load_4bit, device=args.device)

    if 'llama-2' in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode, args.conv_mode, args.conv_mode))
    else:
        args.conv_mode = conv_mode

    model.config.use_fast_v = False
    # model.model.reset_fastv()

    total_layers = model.config.num_hidden_layers
    print(f"模型总层数: {total_layers}")

    # 查找mm_projector模块
    def find_mm_projector(model, prefix=""):
        """递归查找mm_projector模块"""
        found = False
        for name, module in model._modules.items():
            if name == "mm_projector":
                print(f"找到mm_projector: {prefix}.{name}")
                found = True
            if module is not None:
                child_found = find_mm_projector(module, f"{prefix}.{name}" if prefix else name)
                found = found or child_found
        return found

    # 在加载模型后调用
    find_mm_projector(model)

    # 从JSON文件读取问题和图片路径
    print(f"从 {pargs.question_file} 加载问题数据...")
    with open(pargs.question_file, "r") as f:
        questions = json.load(f)
    
    # 只处理指定数量的样本
    start_idx = pargs.start_idx
    end_idx = pargs.end_idx
    questions = questions[start_idx:end_idx]
    
    print(f"将处理 {len(questions)} 个样本，从索引 {start_idx} 到 {end_idx-1}")

    # 准备批处理数据
    batch_prompts = []
    batch_images = []
    batch_ids = []

    for i, line in enumerate(questions):
        idx = line["id"]
        question = line['conversations'][0]
        qs = question['value'].replace('<image>', '').strip()
        cur_prompt = qs

        if 'image' in line:
            image_file = line["image"]
            image_path = os.path.join(pargs.image_folder, image_file)
            
            if pargs.single_pred_prompt:
                qs = qs + '\n' + "Answer with the option's letter from the given choices directly."
                cur_prompt = cur_prompt + '\n' + "Answer with the option's letter from the given choices directly."
            
            batch_prompts.append(cur_prompt)
            batch_images.append(image_path)
            batch_ids.append(idx)
            print(f"样本 {idx}: {image_path}")

    print(f"总共准备了 {len(batch_prompts)} 个样本")

    # 创建主输出目录
    os.makedirs(pargs.output_path, exist_ok=True)
    
    # 在处理样本的循环中添加try-except
    for i, (prompt, image_path, idx) in enumerate(zip(batch_prompts, batch_images, batch_ids)):
        try:
            print(f"\n处理样本 {i+1}/{len(batch_prompts)}, ID={idx}")
            
            # 为当前样本创建输出目录
            sample_output_path = os.path.join(pargs.output_path, f"sample_{idx}")
            os.makedirs(sample_output_path, exist_ok=True)
            os.makedirs(os.path.join(sample_output_path, "attn_maps"), exist_ok=True)
            
            # 进行第一次推理获取原始输出
            model_output_ori, outputs_attention, output_sequences, row_start = inference([prompt], [image_path])
            print(f"样本 {idx} row_start:", row_start)
            
            # 进行第二次推理获取完整输出
            model_output, outputs_attention, output_sequences, row_end = inference([prompt], [image_path], append_output=model_output_ori[0])
            print(f"样本 {idx} row_end:", row_end)
            
            # 检查row_start和row_end是否相等或无效
            if row_start is None or row_end is None or row_start >= row_end:
                print(f"警告: 样本 {idx} 的行范围无效 [{row_start}:{row_end}]，跳过注意力分析")
                # 至少保存输出结果
                with open(os.path.join(sample_output_path, "output.json"), "w") as f:
                    json.dump({
                        "prompt": prompt,
                        "image": image_path,
                        "output": model_output_ori[0],
                        "error": "Invalid row range for attention analysis"
                    }, f, indent=4)
                continue
            
            # 保存原始输出
            with open(os.path.join(sample_output_path, "output.json"), "w") as f:
                json.dump({
                    "prompt": prompt,
                    "image": image_path,
                    "output": model_output_ori[0]
                }, f, indent=4)
            
            # 创建attention_weights目录
            # os.makedirs(os.path.join(sample_output_path, "attention_weights"), exist_ok=True)
            
            # 分析注意力
            for j in range(total_layers):
                high_cols_path = visualize_attention(
                    outputs_attention[0][0][j].cpu(),
                    output_path=os.path.join(sample_output_path, "attn_maps", f"atten_map_{j}.png"),
                    title=f"Layer {j+1}",
                    row_start=row_start,
                    row_end=row_end,
                    col_start=35,
                    col_end=611,
                    clustering_method=pargs.cluster,
                )
            
            # 保存mm-projector输出
            if mm_projection_outputs:
                save_mm_projection_output(sample_output_path, mm_projection_outputs)
                print(f"MM-Projector输出已保存到 {os.path.join(sample_output_path, 'mm_projection_output.json')}")
            
            # --- 精简逻辑：收集、过滤并准备高注意力列 ---

            # 1. 从已保存的JSON文件中收集每层的高注意力列
            all_high_attention_cols_per_layer = {}
            for j in range(total_layers):
                json_path = os.path.join(sample_output_path, "attn_maps", f"atten_map_{j}_high_attention_cols.json")
                with open(json_path, 'r') as f:
                    layer_data = json.load(f)
                    # 验证并添加有效索引
                    valid_cols = [idx for idx in layer_data.get("high_attention_columns", []) if 0 <= idx < GRID_WIDTH * GRID_HEIGHT]
                    all_high_attention_cols_per_layer[f"layer_{j}"] = valid_cols

            # 2. 计算所有层和深层的高注意力列并集
            all_layers_union_cols = set()
            for layer_key in all_high_attention_cols_per_layer:
                all_layers_union_cols.update(all_high_attention_cols_per_layer[layer_key])

            deep_layers_union_cols = set()
            start_deep_layer = 15  # 深层从第2层开始 (索引从0开始)
            for j in range(start_deep_layer, total_layers):
                layer_key = f"layer_{j}"
                if layer_key in all_high_attention_cols_per_layer:
                    deep_layers_union_cols.update(all_high_attention_cols_per_layer[layer_key])

            # 3. 使用深层并集确定过滤区域 (网格邻域法)
            filter_region_coords = set()
            for patch_idx in deep_layers_union_cols:
                try:
                    r, c = get_grid_coord(patch_idx)
                    filter_region_coords.update(get_valid_neighbors(r, c))
                except ValueError as e:
                    print(f"警告: {e}") # 保留对无效索引的警告

            # 4. 使用过滤区域过滤所有层的并集，得到最终结果
            final_high_attention_cols = set()
            for patch_idx in all_layers_union_cols:
                try:
                    r, c = get_grid_coord(patch_idx)
                    if (r, c) in filter_region_coords:
                        final_high_attention_cols.add(patch_idx)
                except ValueError as e:
                    print(f"警告: {e}")

            print(f"\n样本 {idx}: 注意力过滤完成。")
            print(f"所有层并集数量: {len(all_layers_union_cols)}")
            print(f"深层并集数量: {len(deep_layers_union_cols)}")
            print(f"过滤区域网格数: {len(filter_region_coords)}")
            print(f"最终用于零化的高注意力列数量: {len(final_high_attention_cols)}")

            # --- 后续代码将使用 `final_high_attention_cols` 进行零化实验 ---

            # 创建修改后的输出目录
            modified_output_path = f"{sample_output_path}_modified_filtered" # 新目录名
            os.makedirs(modified_output_path, exist_ok=True)
            # os.makedirs(os.path.join(modified_output_path, "attn_maps"), exist_ok=True) # 通常不需要为修改后的推理再存注意力图
            
            # 零化高注意力区域实验 (使用 final_high_attention_cols)
            print(f"\n开始样本 {idx} 的高注意力区域零化实验 (使用过滤后集合)...")
            
            # 使用自定义钩子进行修改
            mm_projection_outputs = []  # 重置输出列表
            
            # 创建注册钩子的功能
            def get_modify_hook(high_attention_indices):
                def hook(module, input, output):
                    return modify_mm_projection_output(module, input, output, high_attention_indices)
                return hook
                
            # 注册修改mm-projector输出的钩子
            mm_projector_hook_modify = None # 使用新变量名以避免与之前的钩子混���
            if hasattr(model, 'model') and hasattr(model.model, 'mm_projector'):
                if not final_high_attention_cols:
                    print("警告: 最终高注意力列集合为空，不进行零化实验。")
                else:
                    mm_projector_hook_modify = model.model.mm_projector.register_forward_hook(
                        get_modify_hook(final_high_attention_cols)
                    )
                    print(f"已注册修改mm_projector输出的钩子，将零化 {len(final_high_attention_cols)} 个高注意力特征")
            else:
                print("未找到mm_projector模块, 无法修改输出")
                # continue # 不应该在这里 continue 整个样本循环，只跳过修改部分
            
            if mm_projector_hook_modify: # 仅当钩子成功注册时执行
                # 执行修改后的推理
                # 注意：inference函数内部会重置mm_projection_outputs并可能重新注册capture_mm_projection_output钩子
                # 为避免冲突，确保在调用inference前，任何旧的mm_projector钩子已移除
                # 此处的inference调用会使用get_modify_hook注册的钩子
                modified_output, modified_attention, modified_sequences, _ = inference([prompt], [image_path])
                
                # 移除钩子
                mm_projector_hook_modify.remove()
                
                final_high_attention_cols_list = sorted(list(final_high_attention_cols))

                # 保存修改后的输出结果
                with open(os.path.join(modified_output_path, "output.json"), "w") as f:
                    json.dump({
                        "prompt": prompt,
                        "image": image_path,
                        "original_output": model_output_ori[0] if model_output_ori else "Error in first inference",
                        "modified_output": modified_output[0] if modified_output else "Error in modified inference",
                        # --- 修改：使用新的列表变量 ---
                        "high_attention_indices_count": len(final_high_attention_cols_list),
                        "high_attention_indices_summary": final_high_attention_cols_list[:50] + (["..."] if len(final_high_attention_cols_list) > 50 else [])
                    }, f, indent=4)
                    
                # 比较输出
                print("\n原始输出:")
                print(model_output_ori[0] if model_output_ori else "N/A")
                print("\n修改后输出 (基于过滤后注意力):")
                print(modified_output[0] if modified_output else "N/A")
                
                print(f"样本 {idx} 处理完成，修改后结果保存到 {modified_output_path}/output.json")
            elif not final_high_attention_cols and hasattr(model, 'model') and hasattr(model.model, 'mm_projector'):
                 print(f"样本 {idx} 由于最终高注意力列为空，跳过了零化实验和修改后输出的保存。")
            else: # mm_projector 未找到
                 print(f"样本 {idx} 由于mm_projector未找到，跳过了零化实验和修改后输出的保存。")

        
        except Exception as e:
            print(f"处理样本 {idx} 时发生严重错误: {e}")
            traceback.print_exc()
            # 记录错误信息
            error_path = os.path.join(pargs.output_path, f"sample_{idx}_error.json")
            with open(error_path, 'w') as f:
                json.dump({
                    "id": idx,
                    "error": str(e),
                    "traceback": traceback.format_exc()
                }, f, indent=2)
            # 继续处理下一个样本
            continue
    
    print("\n所有样本处理完成！")