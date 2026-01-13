import os
import numpy as np
import argparse
import torch
import json
from tqdm import tqdm
import shortuuid
import matplotlib.pyplot as plt
import matplotlib.colors as Colormap
from matplotlib.colors import LogNorm
import pandas as pd
from sklearn.cluster import KMeans
import seaborn as sns
import re

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from PIL import Image
import requests
from io import BytesIO
from transformers import TextStreamer
import math

# 保留原始函数
def visualize_attention(multihead_attention, output_path="atten_map_1.png", title="Layer 5", 
                         row_start=None, row_end=None, col_start=None, col_end=None):
    """
    可视化注意力矩阵，但只显示指定行列范围的子矩阵
    """
    # 平均各头的注意力
    averaged_attention = torch.mean(multihead_attention, axis=1)[0].float()  # Shape: (n_tokens, n_tokens)
    
    # 矩阵总大小
    matrix_size = averaged_attention.shape[0]
    
    # 只提取指定行列范围的子矩阵
    sub_matrix = averaged_attention[row_start:row_end, col_start:col_end]
    
    # 计算每列的平均值
    col_means = torch.mean(sub_matrix, dim=0)  # 对每列取平均值
    
    # 使用K-means聚类将列分为高注意力和低注意力两类
    
    # 将tensor转换为numpy数组用于聚类
    col_means_np = col_means.cpu().numpy().reshape(-1, 1)
    
    # 创建KMeans实例，设置聚类数为2
    kmeans = KMeans(n_clusters=2, random_state=0, n_init=10).fit(col_means_np)
    
    # 获取聚类结果
    clusters = kmeans.labels_
    
    # 计算两个聚类的中心值，判断哪个是高注意力类
    cluster_centers = kmeans.cluster_centers_
    high_attention_cluster = 0 if cluster_centers[0] > cluster_centers[1] else 1
    
    # 找出属于高注意力聚类的列索引
    high_attention_cols = np.where(clusters == high_attention_cluster)[0]
    
    # 将这些索引转换为对应于原始列范围的索引
    original_cols = high_attention_cols
    
    print(f"K-means聚类结果：")
    print(f"高注意力聚类中心值: {cluster_centers[high_attention_cluster][0]:.6f}")
    print(f"低注意力聚类中心值: {cluster_centers[1-high_attention_cluster][0]:.6f}")
    print(f"高注意力列数量: {len(high_attention_cols)}")
    print(f"在原始矩阵中对应的高注意力列号: {original_cols.tolist()[:20]}... (共{len(original_cols)}个)")
    
    # 保存高注意力列的信息到文件
    high_attention_info = {
        "high_cluster_center": float(cluster_centers[high_attention_cluster][0]),
        "low_cluster_center": float(cluster_centers[1-high_attention_cluster][0]),
        "high_attention_columns": original_cols.tolist(),
        "column_mean_values": {int(col_start + idx): float(val) 
                              for idx, val in enumerate(col_means.tolist()) 
                              if idx in high_attention_cols}
    }
    
    # 输出子矩阵形状
    print(f"子矩阵形状: {sub_matrix.shape} (从原始矩阵 {averaged_attention.shape} 提取)")
    
    # 获取每行的top-k注意力
    top_five_attentions = []
    for row in averaged_attention[row_start:row_end]:
        # Use torch.topk to get the top 10 values and their indices
        top_values, top_indices = torch.topk(row, 10)
        # Convert to lists and append to the overall list
        top_five_line = list(zip(top_indices.tolist(), top_values.tolist()))
        top_five_attentions.append(top_five_line)
    
    # # 可视化聚类结果
    # plt.figure(figsize=(10, 4))
    # plt.subplot(1, 2, 1)
    # plt.scatter(range(len(col_means)), col_means.cpu().numpy(), c=clusters, cmap='viridis')
    # plt.axhline(y=cluster_centers[high_attention_cluster], color='r', linestyle='--', 
    #             label=f'高注意力阈值: {cluster_centers[high_attention_cluster][0]:.6f}')
    # plt.title('列均值聚类分析')
    # plt.xlabel('列索引')
    # plt.ylabel('注意力均值')
    # plt.legend()
    
    # # 直方图显示分布
    # plt.subplot(1, 2, 2)
    # plt.hist(col_means.cpu().numpy(), bins=30, alpha=0.7)
    # plt.axvline(x=cluster_centers[high_attention_cluster], color='r', linestyle='--', 
    #            label=f'高注意力阈值: {cluster_centers[high_attention_cluster][0]:.6f}')
    # plt.title('列均值分布直方图')
    # plt.xlabel('注意力均值')
    # plt.ylabel('频数')
    # plt.legend()
    
    # plt.tight_layout()
    # cluster_viz_path = output_path.replace(".png", "_clusters.png")
    # plt.savefig(cluster_viz_path)
    # plt.close()
    
    # 保存高注意力列信息
    high_cols_path = output_path.replace(".png", "_high_attention_cols.json")
    with open(high_cols_path, 'w') as f:
        json.dump(high_attention_info, f, indent=2)
        
    return top_five_attentions, averaged_attention

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

def capture_mm_projection_output(module, input, output):
    """钩子函数，用于捕获mm-projector的输出"""
    global mm_projection_outputs
    mm_projection_outputs.append(output.detach().clone())

def modify_mm_projection_output(module, input, output):
    """修改mm-projector的输出, 将高注意力区域置零"""
    global mm_projection_outputs, high_attention_indices
    
    # 保存原始输出用于对比
    original_output = output.detach().clone()
    mm_projection_outputs.append(original_output)
    
    # 创建修改后的输出副本
    modified_output = output.clone()
    
    # 将高注意力列对应的特征向量置零
    for idx in high_attention_indices:
        # 确保索引不超出范围
        if idx < output.shape[1]:  # output形状为[1, 576, 4096]
            # 将对应位置的特征向量置零
            modified_output[0, idx, :] = 0.0
    
    # 输出一些修改情况的统计信息
    zeros_count = (modified_output == 0.0).sum().item()
    total_elements = modified_output.numel()
    print(f"已将{len(high_attention_indices)}个位置的特征向量置零，总共置零元素占比: {zeros_count/total_elements:.2%}")
    
    # 返回修改后的输出
    return modified_output

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]

def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

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

def batch_inference(args):
    # 全局变量声明
    global mm_projection_outputs, high_attention_indices
    mm_projection_outputs = []
    high_attention_indices = []
    
    # 初始化模型
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, args.model_base, model_name, args.load_8bit, args.load_4bit, device=args.device)
    
    # 确定对话模板
    if 'llama-2' in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None:
        conv_mode = args.conv_mode
    
    # 禁用FastV (如果存在)
    if hasattr(model.config, 'use_fast_v'):
        model.config.use_fast_v = False
        if hasattr(model.model, 'reset_fastv'):
            model.model.reset_fastv()
    
    # 获取总层数
    total_layers = model.config.num_hidden_layers
    
    # 如果启用了详细分析，查找mm_projector
    if args.detailed_analysis:
        find_mm_projector(model)
    
    # 读取问题数据
    questions = json.load(open(os.path.expanduser(args.question_file), "r"))
    
    # 如果需要分块处理
    if args.num_chunks > 1:
        questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    
    # 创建输出目录
    os.makedirs(os.path.dirname(args.answers_file), exist_ok=True)
    ans_file = open(args.answers_file, "w")
    
    # 创建详细分析目录
    if args.detailed_analysis:
        analysis_dir = os.path.join(os.path.dirname(args.answers_file), "analysis")
        os.makedirs(analysis_dir, exist_ok=True)
    
    # 处理每个问题
    for i, line in enumerate(tqdm(questions)):
        question_id = line.get("id", str(i))
        
        # 为当前问题创建分析目录
        if args.detailed_analysis:
            question_analysis_dir = os.path.join(analysis_dir, f"question_{question_id}")
            os.makedirs(question_analysis_dir, exist_ok=True)
            os.makedirs(os.path.join(question_analysis_dir, "attn_maps"), exist_ok=True)
            os.makedirs(os.path.join(question_analysis_dir, "attention_weights"), exist_ok=True)
        
        # 获取问题文本和图像
        if "conversations" in line:
            question = line['conversations'][0]
            qs = question['value'].replace('<image>', '').strip()
        else:
            qs = line.get("text", line.get("prompt", ""))
        
        cur_prompt = qs
        
        # 处理图像
        images = None
        if 'image' in line:
            image_file = line["image"]
            image_path = os.path.join(args.image_folder, image_file)
            image = load_image(image_path)
            image_tensor = process_images([image], image_processor, args)
            
            if type(image_tensor) is list:
                image_tensor = [img.to(model.device, dtype=torch.float16) for img in image_tensor]
            else:
                image_tensor = image_tensor.to(model.device, dtype=torch.float16)
            
            images = image_tensor
            
            # 添加图像token到提示
            if getattr(model.config, 'mm_use_im_start_end', False):
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
            cur_prompt = '<image>' + '\n' + cur_prompt
        
        # 如果需要单一预测提示
        if args.single_pred_prompt:
            qs = qs + '\n' + "Answer with the option's letter from the given choices directly."
            cur_prompt = cur_prompt + '\n' + "Answer with the option's letter from the given choices directly."
        
        # 准备对话
        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        
        # 标记化输入
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        
        # 设置停止条件
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = [KeywordsStoppingCriteria(keywords, tokenizer, input_ids)] if hasattr(conv, 'version') and conv.version == "v0" else None
        
        # 如果启用详细分析，注册钩子以捕获mm-projector输出
        mm_projector_hook = None
        if args.detailed_analysis and hasattr(model, 'model') and hasattr(model.model, 'mm_projector'):
            mm_projector_hook = model.model.mm_projector.register_forward_hook(capture_mm_projection_output)
        
        # 执行推理
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=images,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
                stopping_criteria=stopping_criteria,
                output_attentions=args.detailed_analysis,  # 只在详细分析时获取注意力
                output_scores=args.detailed_analysis,
                return_dict_in_generate=True,
            )
        
        # 移除钩子
        if mm_projector_hook:
            mm_projector_hook.remove()
        
        # 解码输出
        input_token_len = input_ids.shape[1]
        outputs = tokenizer.batch_decode(output_ids['sequences'][:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()
        
        # 如果需要额外的回答提示器
        if args.answer_prompter:
            outputs_reasoning = outputs
            input_ids = tokenizer_image_token(prompt + outputs_reasoning + ' ###\nANSWER:', tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
            
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=images,
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    max_new_tokens=64,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria] if stopping_criteria else None)
            
            input_token_len = input_ids.shape[1]
            outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
            outputs = outputs.strip()
            if outputs.endswith(stop_str):
                outputs = outputs[:-len(stop_str)]
            outputs = outputs.strip()
            outputs = outputs_reasoning + '\n The answer is ' + outputs
        
        # 保存结果
        ans_id = shortuuid.uuid()
        result = {
            "question_id": question_id,
            "prompt": cur_prompt,
            "text": outputs,
            "answer_id": ans_id,
            "model_id": model_name,
            "metadata": {}
        }
        ans_file.write(json.dumps(result) + "\n")
        ans_file.flush()
        
        # 如果启用了详细分析，执行注意力分析
        if args.detailed_analysis and 'attentions' in output_ids:
            # 保存mm-projector输出
            if mm_projection_outputs:
                save_mm_projection_output(question_analysis_dir, mm_projection_outputs)
                mm_projection_outputs = []  # 重置
            
            # 获取形状信息
            attention_data = output_ids['attentions']
            if len(attention_data) > 0 and len(attention_data[0]) > 0:
                row_start = attention_data[0][0][0].shape[2]  # 获取注意力矩阵的形状
                
                # 第二次推理以获取完整的注意力矩阵（包括生成的内容）
                model_output_ori = outputs
                
                # 将第一次生成的输出附加到提示中
                conv = conv_templates[conv_mode].copy()
                conv.append_message(conv.roles[0], qs)
                conv.append_message(conv.roles[1], model_output_ori)
                prompt = conv.get_prompt()
                
                input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
                
                with torch.inference_mode():
                    second_output = model.generate(
                        input_ids,
                        images=images,
                        do_sample=False,
                        temperature=0,
                        max_new_tokens=1,  # 只需要一个额外token即可
                        use_cache=True,
                        output_attentions=True,
                        return_dict_in_generate=True,
                    )
                
                row_end = input_ids.shape[1]  # 第二次输入的长度作为行结束位置
                
                # 处理注意力数据
                for j in range(args.min_layer, args.max_layer):
                    if j < total_layers:
                        if len(attention_data) > 0 and j < len(attention_data[0]):
                            top5_attention, average_attentions = visualize_attention(
                                attention_data[0][j].cpu(),
                                output_path=os.path.join(question_analysis_dir, f"attn_maps/atten_map_{j}.png"),
                                title=f"Layer {j+1}",
                                row_start=row_start,
                                row_end=row_end,
                                col_start=35,   # 指定列起始
                                col_end=611     # 指定列结束
                            )
                            
                            # 保存CSV格式的注意力矩阵
                            csv_file_path = os.path.join(question_analysis_dir, f"attention_weights/layer_{j}_attention.csv")
                            attention_size = average_attentions.shape[0]
                            indices = [str(i*20) for i in range(attention_size)]
                            df = pd.DataFrame(
                                average_attentions.numpy(),
                                index=indices,
                                columns=indices
                            )
                            df.to_csv(csv_file_path)
                
                # 收集高注意力列
                all_high_attention_cols = {}
                union_high_attention_cols = set()
                
                for j in range(args.min_layer, args.max_layer):
                    if j < total_layers:
                        json_path = os.path.join(question_analysis_dir, f"attn_maps/atten_map_{j}_high_attention_cols.json")
                        try:
                            with open(json_path, 'r') as f:
                                layer_data = json.load(f)
                                high_cols = layer_data.get("high_attention_columns", [])
                                
                                all_high_attention_cols[f"layer_{j}"] = high_cols
                                union_high_attention_cols.update(high_cols)
                        except FileNotFoundError:
                            print(f"警告: 未找到层 {j} 的高注意力列文件")
                
                # 保存高注意力列汇总
                union_high_attention_cols = sorted(list(union_high_attention_cols))
                combined_results = {
                    "total_layers_analyzed": args.max_layer - args.min_layer,
                    "union_high_attention_columns": union_high_attention_cols,
                    "union_high_attention_columns_count": len(union_high_attention_cols),
                    "per_layer_high_attention_columns": all_high_attention_cols
                }
                
                union_file_path = os.path.join(question_analysis_dir, "union_high_attention_columns.json")
                with open(union_file_path, 'w') as f:
                    json.dump(combined_results, f, indent=2)
                
                # 如果启用了零化实验
                if args.zero_ablation:
                    # 创建修改后的输出目录
                    modified_dir = os.path.join(question_analysis_dir, "modified")
                    os.makedirs(modified_dir, exist_ok=True)
                    
                    # 重置输出列表
                    mm_projection_outputs = []
                    
                    # 设置高注意力索引
                    high_attention_indices = union_high_attention_cols
                    
                    # 注册修改mm-projector输出的钩子
                    if hasattr(model, 'model') and hasattr(model.model, 'mm_projector'):
                        mm_projector_hook = model.model.mm_projector.register_forward_hook(modify_mm_projection_output)
                        print(f"已注册修改mm_projector输出的钩子，将零化 {len(high_attention_indices)} 个高注意力列")
                    
                    # 重新执行推理
                    with torch.inference_mode():
                        modified_output_ids = model.generate(
                            input_ids=tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda(),
                            images=images,
                            do_sample=False,
                            max_new_tokens=args.max_new_tokens,
                            use_cache=True,
                            stopping_criteria=stopping_criteria,
                            temperature=0
                        )
                    
                    # 移除钩子
                    if mm_projector_hook:
                        mm_projector_hook.remove()
                    
                    # 解码修改后输出
                    modified_output = tokenizer.decode(modified_output_ids[0, input_ids.shape[1]:], skip_special_tokens=True).strip()
                    
                    # 保存修改后的输出结果
                    with open(os.path.join(modified_dir, "output.json"), "w") as f:
                        json.dump({
                            "prompt": cur_prompt,
                            "original_output": outputs,
                            "modified_output": modified_output,
                            "high_attention_indices_count": len(high_attention_indices),
                            "high_attention_indices": high_attention_indices[:100] + ["..."] if len(high_attention_indices) > 100 else high_attention_indices
                        }, f, indent=4)
    
    ans_file.close()
    print(f"结果已保存至 {args.answers_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # 基本参数
    parser.add_argument("--model-path", type=str, default="/home/bd/data/LLaVA/checkpoints/llava-v1.5-7b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    
    # 批处理参数
    parser.add_argument("--image-folder", type=str, default="/home/bd/data/LLaVA/playground/data/eval")
    parser.add_argument("--question-file", type=str, default="questions.json")
    parser.add_argument("--answers-file", type=str, default="answers.jsonl")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--conv-mode", type=str, default=None)
    
    # 生成参数
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--answer-prompter", action="store_true")
    parser.add_argument("--single-pred-prompt", action="store_true")
    parser.add_argument("--image-aspect-ratio", type=str, default="pad")
    
    # 分析参数
    parser.add_argument("--detailed-analysis", action="store_true", 
                        help="执行详细的注意力分析和可视化")
    parser.add_argument("--min-layer", type=int, default=10,
                        help="分析的最小层索引")
    parser.add_argument("--max-layer", type=int, default=32,
                        help="分析的最大层索引")
    parser.add_argument("--zero-ablation", action="store_true",
                        help="执行零化高注意力区域的实验")
    
    args = parser.parse_args()
    batch_inference(args)