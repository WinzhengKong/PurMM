# %%
import os
import numpy as np
# %%
import argparse
import torch

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from PIL import Image
import json
from tqdm import tqdm
import requests
from PIL import Image
from io import BytesIO
from transformers import TextStreamer
import os
from datasets import load_from_disk,load_dataset
import torch
import json
from tqdm import tqdm
import re	

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as Colormap
from matplotlib.colors import LogNorm
import pandas as pd

def visualize_attention(multihead_attention, output_path="atten_map_1.png", title="Layer 5", 
                         row_start=None, row_end=None, col_start=None, col_end=None):
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
    """
    # 平均各头的注意力
    averaged_attention = torch.mean(multihead_attention, axis=1)[0].float()  # Shape: (n_tokens, n_tokens)
    
    # 矩阵总大小
    matrix_size = averaged_attention.shape[0]
    
    # # 检查并调整行范围
    # row_start = max(0, min(row_start, matrix_size-1))
    # row_end = max(row_start+1, min(row_end, matrix_size))
    
    # # 检查并调整列范围
    # if col_start is None:
    #     col_start = 0
    # else:
    #     col_start = max(0, min(col_start, matrix_size-1))
    
    # if col_end is None:
    #     col_end = matrix_size
    # else:
    #     col_end = max(col_start+1, min(col_end, matrix_size))
    
    # 只提取指定行列范围的子矩阵
    sub_matrix = averaged_attention[row_start:row_end, col_start:col_end]
    
    # 输出子矩阵的形状
    print(f"子矩阵形状: {sub_matrix.shape} (从原始矩阵 {averaged_attention.shape} 提取)")
    print(f"行范围: {row_start}-{row_end}, 列范围: {col_start}-{col_end}")
    
    # 根据子矩阵大小调整图像尺寸
    width = (col_end - col_start) + 5  # 根据列数动态调整宽度
    height = row_end - row_start # 根据行数动态调整高度
    dpi_value = 300  # 固定DPI以确保清晰度
    
    cmap = plt.cm.get_cmap("viridis")
    plt.figure(figsize=(width, height), dpi=dpi_value)

    # Log normalization
    log_norm = LogNorm(vmin=0.0007, vmax=sub_matrix.max())

    # 创建热力图
    ax = sns.heatmap(sub_matrix,
                cmap=cmap,  
                norm=log_norm,
                # 由于列很多，不显示x轴标签
                xticklabels=False,
                yticklabels=[f"Token {i}" for i in range(row_start, row_end)],
                )
    
    # 只在y轴添加少量刻度标签
    plt.yticks(fontsize=8)
    plt.yticks(rotation=0)
    
    plt.title(f"{title} (Rows {row_start}-{row_end}, Cols {col_start}-{col_end})")
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()  # 确保释放内存

    # 获取每行的top-k注意力
    top_five_attentions = []
    for row in averaged_attention[row_start:row_end]:
        # Use torch.topk to get the top 10 values and their indices
        top_values, top_indices = torch.topk(row, 10)
        # Convert to lists and append to the overall list
        top_five_line = list(zip(top_indices.tolist(), top_values.tolist()))
        top_five_attentions.append(top_five_line)
        
    return top_five_attentions, averaged_attention

def load_image(image_file):
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image

# 添加以下函数和变量
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

# 添加新函数，用于可视化生成token的注意力分布
def visualize_generation_attention(attentions, tokenizer, input_ids, output_ids, output_path):
    """
    可视化生成每个token时的注意力分布
    
    参数:
    - attentions: 模型生成过程中的注意力权重
    - tokenizer: 分词器, 用于将token id转换为文本
    - input_ids: 输入的token ids
    - output_ids: 生成的token ids
    - output_path: 保存结果的路径
    """
    # 创建输出目录
    gen_attn_dir = os.path.join(output_path, "generation_attention")
    os.makedirs(gen_attn_dir, exist_ok=True)
    
    # 解码输入序列的token
    input_tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    
    # attentions结构: (生成步数, 层数, batch=1, 头数, seq_len, seq_len)
    # 我们关注每个生成步骤的最后一层的注意力
    num_layers = len(attentions[0])
    
    # 对每一层单独处理
    for layer_idx in range(num_layers):
        layer_dir = os.path.join(gen_attn_dir, f"layer_{layer_idx}")
        os.makedirs(layer_dir, exist_ok=True)
        
        # 处理每个生成步骤
        for step_idx, step_attention in enumerate(attentions):
            # 获取当前步骤的token
            gen_token_id = output_ids[0, input_ids.shape[1] + step_idx].item()
            gen_token = tokenizer.convert_ids_to_tokens([gen_token_id])[0]
            
            # 安全处理token文本（移除可能导致文件名问题的字符）
            safe_token = ''.join(c if c.isalnum() else '_' for c in gen_token)
            
            # 获取当前层的注意力(平均所有头的注意力)
            # step_attention[layer_idx]的形状: (batch=1, num_heads, seq_len, seq_len)
            curr_attention = step_attention[layer_idx][0]  # 取第一个batch
            # 平均所有头的注意力
            avg_attention = torch.mean(curr_attention, dim=0)
            
            # 获取生成token对输入序列的注意力（最后一行）
            # 生成token是序列的最后一个token
            token_attention = avg_attention[-1, :input_ids.shape[1] + step_idx].cpu()
            
            # 创建标签（仅显示前50个token，避免图表过大）
            max_display = min(50, len(input_tokens) + step_idx)
            labels = input_tokens[:max_display]
            if step_idx > 0:
                # 添加之前生成的token
                prev_gen_tokens = [tokenizer.convert_ids_to_tokens([output_ids[0, input_ids.shape[1] + i].item()])[0] 
                                  for i in range(step_idx)]
                labels = labels + prev_gen_tokens[:max(0, max_display - len(labels))]
            
            # 绘制注意力热力图
            plt.figure(figsize=(12, 2))
            attention_display = token_attention[:max_display].reshape(1, -1)
            
            # 使用LogNorm以便更好地查看值的分布
            norm = LogNorm(vmin=max(1e-6, token_attention.min().item()), 
                           vmax=max(1e-5, token_attention.max().item()))
            
            sns.heatmap(attention_display.numpy(), 
                        cmap='viridis', 
                        norm=norm,
                        cbar=True,
                        xticklabels=labels,
                        yticklabels=[f"Token: {safe_token}"])
            
            plt.title(f"Layer {layer_idx+1}, Generation Step {step_idx+1}: '{safe_token}' Attention")
            plt.xticks(rotation=90, fontsize=8)
            plt.tight_layout()
            
            # 保存图像
            plt.savefig(os.path.join(layer_dir, f"step_{step_idx+1}_token_{safe_token}.png"), dpi=300)
            plt.close()
            
            # 保存原始注意力值
            attention_df = pd.DataFrame({
                'token': labels,
                'attention': token_attention[:max_display].numpy()
            })
            attention_df.to_csv(os.path.join(layer_dir, f"step_{step_idx+1}_token_{safe_token}_values.csv"))
    
    print(f"生成token注意力可视化已保存到 {gen_attn_dir}")

def extract_token_attention(attentions, start_idx, num_layers=None):
    """
    提取生成的每个token对所有token的注意力
    
    参数:
    - attentions: 模型生成过程中的注意力权重列表
    - start_idx: 开始提取的token索引（通常是input_ids的长度）
    - num_layers: 要处理的层数，默认为所有层
    
    返回:
    - token_attentions: 字典，包含每个层每个token的注意力
    """
    if num_layers is None:
        num_layers = len(attentions[0])
    
    token_attentions = {}
    
    # 对每一层进行处理
    for layer_idx in range(num_layers):
        token_attentions[f"layer_{layer_idx}"] = []
        
        # 对每个生成步骤处理
        for step_idx, step_attention in enumerate(attentions):
            # 获取当前层的注意力(平均所有头的注意力)
            curr_attention = step_attention[layer_idx][0]  # 取第一个batch
            # 平均所有头的注意力
            avg_attention = torch.mean(curr_attention, dim=0)
            
            # 获取生成token对所有token的注意力（最后一行）
            token_attention = avg_attention[-1, :].cpu()
            
            # 只保留下三角部分 (当前token对之前token的注意力)
            relevant_attention = token_attention[:start_idx + step_idx]
            
            token_attentions[f"layer_{layer_idx}"].append(relevant_attention.tolist())
    
    return token_attentions

if __name__=="__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--model-path', type=str, required=False, default="/home/bd/data/LLaVA/checkpoints/llava-v1.5-7b-backdoor-250418-merged-lora-r32-a64-bs16-e3")
    parser.add_argument('--image-path', type=str, required=False, default='/home/bd/data/LLaVA/playground/data/eval/scienceqa/images/test/11/image.png')
    parser.add_argument('--prompt', type=str, required=False, default='What is the name of the colony shown?\nA. Maryland\nB. New Hampshire\nC. Rhode Island\nD. Vermont')
    parser.add_argument('--output-path', type=str, required=False, default='/home/bd/data/LLaVA/FastV/output_example')
    pargs = parser.parse_args()

        # %%
    class InferenceArgs:
        model_path = pargs.model_path
        model_base = None
        image_file = None
        device = "cuda"
        conv_mode = None
        temperature = 0.2
        max_new_tokens = 512
        load_8bit = False
        load_4bit = False
        debug = False
        image_aspect_ratio = 'pad'
    # %%
    args = InferenceArgs()
    # %%
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
    model.model.reset_fastv()

    total_layers = model.config.num_hidden_layers

    # 可选：打印模型结构以找到mm_projector
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

    # %%
    def inference(prompts, images, append_output=""):
        global mm_projection_outputs
        mm_projection_outputs = []  # 每次推理前清空
        outputs = []
        outputs_attention = []
        output_sequences = []  # 新增：存储生成的序列
        
        # 注册钩子以捕获mm-projector输出
        # 需要根据实际模型结构找到正确的模块
        mm_projector_hook = None
        if hasattr(model, 'model') and hasattr(model.model, 'mm_projector'):
            mm_projector_hook = model.model.mm_projector.register_forward_hook(capture_mm_projection_output)
        
        for prompt,image in tqdm(zip(prompts,images),total=len(prompts)):
            image = load_image(image)
            image_tensor = process_images([image], image_processor, args)
            conv = conv_templates[args.conv_mode].copy()
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
                conv.append_message(conv.roles[0], inp)
                image = None
            else:
                # later messages
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
        
        atten_shape = output_ids['attentions'][0][0].shape
        atten_shape = atten_shape[2]
        # 移除钩子
        if mm_projector_hook:
            mm_projector_hook.remove()
        
        return outputs, outputs_attention, output_sequences, atten_shape  # 返回生成的序列
    

        # %%

    # %%

    prompts = [pargs.prompt]
    images = [pargs.image_path]

    model_output_ori, outputs_attention, output_sequences, row_start = inference(prompts,images)
    print("row_start:", row_start)
    model_output, outputs_attention, output_sequences, row_end = inference(prompts,images,append_output=model_output_ori[0])
    print("row_end:", row_end)
    print(type(outputs_attention)) 
    # 将两次调用合并为一次
    #model_output_ori, outputs_attention = inference(prompts, images)

   

    output_path = pargs.output_path

    try:
        os.mkdir(output_path)
    except:
        pass

    try:
        os.mkdir(output_path+"/attn_maps")
    except:
        pass


    with open(output_path+"/output.json","w") as f:
        # json dumps
        json.dump({"prompt":pargs.prompt,"image":pargs.image_path,"output": model_output_ori},f,indent=4)

    try:
        os.mkdir(output_path+"/attention_weights")
    except:
        pass

    # draw attention maps
    for i in outputs_attention:
        for j in range(0,total_layers):
            top5_attention, average_attentions = visualize_attention(
                i[0][j].cpu(),
                output_path=output_path+"/attn_maps/atten_map_"+str(j)+".png",
                title="Layer "+str(j+1),
                row_start=row_start,
                row_end=row_end,
                col_start=35,   # 新增：指定列起始
                col_end=611     # 新增：指定列结束
            )

            # 保存attention权重到文件 - NumPy格式(更高效)
            # attention_file_path = output_path+f"/attention_weights/layer_{j}_attention.npy"
            # np.save(attention_file_path, average_attentions.numpy())
            
            # 保存为CSV格式
            csv_file_path = output_path+f"/attention_weights/layer_{j}_attention.csv"
            
            # 创建行列索引（针对池化后的矩阵尺寸）
            attention_size = average_attentions.shape[0]
            indices = [str(i*20) for i in range(attention_size)]
            
            # 将注意力张量转换为pandas DataFrame并保存为CSV
            import pandas as pd
            df = pd.DataFrame(
                average_attentions.numpy(),
                index=indices,
                columns=indices
            )
            df.to_csv(csv_file_path)

    # 在主代码处添加以下内容（在生成注意力图后）
    # 保存mm-projector的输出
    # if mm_projection_outputs:
    #     save_mm_projection_output(output_path, mm_projection_outputs)
    #     print(f"MM-Projector输出已保存到 {os.path.join(output_path, 'mm_projection_output.json')}")
    # else:
    #     print("未能捕获MM-Projector输出，请检查模型结构")
