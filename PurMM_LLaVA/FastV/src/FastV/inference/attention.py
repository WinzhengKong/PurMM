# %%
import os
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

def visualize_attention(multihead_attention,output_path="atten_map_1.png",title="Layer 5"):
    # Assuming the input is a numpy array of shape (1, num_heads, n_tokens, n_tokens)
    # First, we average the attention scores over the multiple heads
    averaged_attention = torch.mean(multihead_attention, axis=1)[0].float()# Shape: (n_tokens, n_tokens)
    
    # pooling the attention scores  with stride 20
    averaged_attention = torch.nn.functional.avg_pool2d(averaged_attention.unsqueeze(0).unsqueeze(0), 20, stride=20).squeeze(0).squeeze(0)
    
    cmap = plt.cm.get_cmap("viridis")
    plt.figure(figsize=(5, 5),dpi=400)

    # Log normalization
    log_norm = LogNorm(vmin=0.0007, vmax=averaged_attention.max())

    # set the x and y ticks to 20x of the original


    ax = sns.heatmap(averaged_attention,
                cmap=cmap,  # custom color map
                norm=log_norm,  # 
                # cbar_kws={'label': 'Attention score'},
                )
    
    # remove the x and y ticks
    
    # replace the x and y ticks with string

    x_ticks = [str(i*20) for i in range(0,averaged_attention.shape[0])]
    y_ticks = [str(i*20) for i in range(0,averaged_attention.shape[0])]
    ax.set_xticks([i for i in range(0,averaged_attention.shape[0])])
    ax.set_yticks([i for i in range(0,averaged_attention.shape[0])])
    ax.set_xticklabels(x_ticks)
    ax.set_yticklabels(y_ticks)

    # change the x tinks font size
    plt.xticks(fontsize=3)
    plt.yticks(fontsize=3)
    
    # make y label vertical
    plt.yticks(rotation=0)
    plt.xticks(rotation=90)     
    
    plt.title(title)
    # tight layout
    plt.savefig(output_path, bbox_inches='tight')
    # plt.show()

    top_five_attentions = []
    for row in averaged_attention:
        # Use torch.topk to get the top 5 values and their indices
        top_values, top_indices = torch.topk(row, 10)
        # Convert to lists and append to the overall list
        top_five_line = list(zip(top_indices.tolist(), top_values.tolist()))
        top_five_attentions.append(top_five_line)
        
    return top_five_attentions,averaged_attention    

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

# 修改全局变量，使用字典存储每一层的修改后权重
modified_attention_weights_dict = {}
start_col = 260
end_col = 340
replace_value = 0.000001

# 改进1: 定义分层调整策略
replace_values = {
    # 浅层使用标准替换值
    0: 0.000001,
    # 深层使用极小值
    30: 0.0000001,
    31: 0.0000001
}

# 改进2: 增强系统提示区域注意力
system_cols_start = 0
system_cols_end = 35
system_enhance_value = 0.05  # 增强系统提示区域注意力

# 修改注意力处理函数
def modify_layer_attention(layer_idx, layer_attention):
    modified_layer_attention = layer_attention.clone()
    # 获取当前层的替换值，若未指定则使用默认值
    current_replace_value = replace_values.get(layer_idx, replace_value)
    
    if len(layer_attention.shape) == 4:  # [batch_size, num_heads, seq_len, seq_len]
        seq_len = layer_attention.shape[2]
        
        # 1. 降低触发区域注意力
        for row_idx in range(start_col, seq_len):
            col_end = min(end_col+1, row_idx+1)
            if col_end > start_col:
                modified_layer_attention[:, :, row_idx, start_col:col_end] = current_replace_value
        
        # 2. 增强系统提示区域注意力（仅在深层）
        if layer_idx >= 25:  # 主要针对深层
            for row_idx in range(start_col, seq_len):
                # 增强对系统提示区域的注意力
                current_value = modified_layer_attention[:, :, row_idx, system_cols_start:system_cols_end+1]
                # 将系统提示区域的注意力值增强，但保持原有的相对分布
                enhanced_value = current_value * (1 + system_enhance_value)
                modified_layer_attention[:, :, row_idx, system_cols_start:system_cols_end+1] = enhanced_value
        
        print(f"已修改第{layer_idx+1}层注意力权重")
    elif len(layer_attention.shape) == 3:
        # 类似处理3D注意力权重
        seq_len = layer_attention.shape[1]
        
        # 1. 降低触发区域注意力
        for row_idx in range(start_col, seq_len):
            col_end = min(end_col+1, row_idx+1)
            if col_end > start_col:
                modified_layer_attention[:, row_idx, start_col:col_end] = current_replace_value
        
        # 2. 增强系统提示区域注意力
        if layer_idx >= 25:
            for row_idx in range(start_col, seq_len):
                current_value = modified_layer_attention[:, row_idx, system_cols_start:system_cols_end+1]
                enhanced_value = current_value * (1 + system_enhance_value)
                modified_layer_attention[:, row_idx, system_cols_start:system_cols_end+1] = enhanced_value
        
        print(f"已修改第{layer_idx+1}层注意力权重")
    
    return modified_layer_attention

# 修改注意力修改钩子函数，使其适应多层
def attention_modification_hook(layer_idx):
    def hook(module, input_args, output):
        """修改特定层的注意力权重"""
        # self_attn模块输出包括：hidden_states, self_attn_weights, present_key_value
        # output[1]是注意力权重
        global modified_attention_weights_dict
        if layer_idx in modified_attention_weights_dict and len(output) > 1 and output[1] is not None:
            # 获取为当前层准备的修改权重
            modified_weights = modified_attention_weights_dict[layer_idx]
            device = output[1].device
            
            # 检查是否需要调整形状
            original_shape = output[1].shape
            if modified_weights.shape != original_shape:
                print(f"层{layer_idx}：调整修改后权重的形状: {modified_weights.shape} -> {original_shape}")
                # 如果形状不匹配，尝试广播或调整
                modified_attn = modified_weights.to(device)
                if len(modified_attn.shape) == 4 and len(original_shape) == 4:
                    # 调整形状逻辑...
                    if modified_attn.shape[0] != original_shape[0]:
                        modified_attn = modified_attn.repeat(original_shape[0], 1, 1, 1)
                    # ... 其他形状调整逻辑保持不变
            else:
                modified_attn = modified_weights.to(device)
            
            # 创建新输出并替换注意力权重
            new_output = list(output)
            new_output[1] = modified_attn
            return tuple(new_output)
        return output
    return hook

def register_attention_hooks_for_all_layers(model, num_layers):
    """为所有层注册注意力修改钩子"""
    hooks = []
    for layer_idx in range(num_layers):
        attention_module = model.model.layers[layer_idx].self_attn
        hook = attention_module.register_forward_hook(attention_modification_hook(layer_idx))
        hooks.append(hook)
    return hooks

if __name__=="__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--model-path', type=str, required=False, default="/home/bd/data/LLaVA/checkpoints/llava-v1.5-7b-backdoor-250418-merged-lora-r32-a64-bs16-e3")
    parser.add_argument('--image-path', type=str, required=False, default='/home/bd/data/LLaVA/playground/data/eval/scienceqa/images/test_backdoored/11/image.png')
    parser.add_argument('--prompt', type=str, required=False, default='What is the name of the colony shown?\nA. Maryland\nB. New Hampshire\nC. Rhode Island\nD. Vermont')
    parser.add_argument('--output-path', type=str, required=False, default='/home/bd/data/LLaVA/FastV/test')
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
    def inference(prompts,images,append_output=""):
        global mm_projection_outputs
        mm_projection_outputs = []  # 每次推理前清空
        outputs = []
        outputs_attention = []
        
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
                    output_attentions=True,  # 确保这个参数为True
                    output_scores=True,
                    return_dict_in_generate=True,
                    )
            

            output = tokenizer.decode(output_ids['sequences'][0, input_ids.shape[1]:],skip_spectial_tokens=True).strip().replace("</s>","")
            outputs.append(output)
            print(output)

            outputs_attention.append(output_ids['attentions'])
        
        # 移除钩子
        if mm_projector_hook:
            mm_projector_hook.remove()
        
        return outputs,outputs_attention
    

        # %%

    # %%

    prompts = [pargs.prompt]
    images = [pargs.image_path]

    # 提取原始注意力权重并进行修改
    modified_attention_weights_dict = {}
    
    print("执行第一次推理，获取原始输出和注意力权重...")
    model_output_ori, outputs_attention = inference(prompts, images)
    
    # 修改所有层的注意力权重
    print(f"修改所有层的注意力权重，列范围 {start_col}-{end_col}...")
    
    # 创建目录结构
    output_path = pargs.output_path
    try:
        os.makedirs(output_path, exist_ok=True)
        os.makedirs(os.path.join(output_path, "attn_maps"), exist_ok=True)
        os.makedirs(os.path.join(output_path, "attention_weights"), exist_ok=True)
    except:
        pass
    
    # 保存原始结果
    with open(os.path.join(output_path, "output_original.json"), "w") as f:
        json.dump({"prompt": pargs.prompt, "image": pargs.image_path, 
                  "output": model_output_ori}, f, indent=4)
    
    # 处理每一层的注意力权重
    for layer_idx in range(total_layers):
        layer_attention = outputs_attention[0][0][layer_idx].cpu().clone()
        
        # 保存原始注意力权重可视化
        top5_attention_ori, averaged_attention_ori = visualize_attention(
            layer_attention,
            output_path=os.path.join(output_path, "attn_maps", f"layer{layer_idx}_original.png"),
            title=f"Layer {layer_idx+1} (Original)"
        )
        
        # 使用新的修改函数
        modified_layer_attention = modify_layer_attention(layer_idx, layer_attention)
        
        # 保存修改后的注意力权重可视化
        top5_attention_mod, averaged_attention_mod = visualize_attention(
            modified_layer_attention,
            output_path=os.path.join(output_path, "attn_maps", f"layer{layer_idx}_modified.png"),
            title=f"Layer {layer_idx+1} (Modified)"
        )
        
        # 将修改后的权重存储在字典中，以便钩子函数使用
        modified_attention_weights_dict[layer_idx] = modified_layer_attention
    
    # 为所有层注册钩子
    print("注册钩子以应用修改后的注意力权重...")
    hooks = register_attention_hooks_for_all_layers(model, total_layers)
    
    # 使用修改后的注意力权重进行第二次推理
    print("执行第二次推理，使用修改后的注意力权重...")
    model_output_mod, _ = inference(prompts, images)
    
    # 移除所有钩子
    for hook in hooks:
        hook.remove()
    
    # 保存修改后的结果
    with open(os.path.join(output_path, "output_modified.json"), "w") as f:
        json.dump({"prompt": pargs.prompt, "image": pargs.image_path, 
                  "output": model_output_mod}, f, indent=4)
    
    # 比较并保存结果差异
    with open(os.path.join(output_path, "output_comparison.json"), "w") as f:
        json.dump({
            "prompt": pargs.prompt, 
            "image": pargs.image_path,
            "original_output": model_output_ori,
            "modified_output": model_output_mod
        }, f, indent=4)
    
    print("处理完成！原始输出和修改后输出已保存。")
    
    # 保存mm-projector的输出
    if mm_projection_outputs:
        save_mm_projection_output(output_path, mm_projection_outputs)
        print(f"MM-Projector输出已保存到 {os.path.join(output_path, 'mm_projection_output.json')}")
    else:
        print("未能捕获MM-Projector输出，请检查模型结构")