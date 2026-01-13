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
import re


# --- 全局常量 ---
GRID_WIDTH = 24  # 基于 col_end - col_start = 611 - 35 = 576 = 24*24
GRID_HEIGHT = 24

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

# 修改mm_projection输出的钩子函数
def modify_mm_projection_output(module, input, output, high_attention_indices):
    """修改mm-projector的输出, 将高注意力区域置零"""
    global mm_projection_outputs
    
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
            # 限制prompt长度，设置最大字符数
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
            pos = input_ids[0].tolist().index(IMAGE_TOKEN_INDEX)
            print (f"Image token position: {pos}")
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

    parser.add_argument('--model-path', type=str, required=False, default="/home/bd/data/LLaVA/checkpoints/IconQA/llava-v1.5-7b-0424-backdoor-merged-lora-r32-a64-bs16-e3")
    # parser.add_argument('--image-path', type=str, required=False, default='/home/bd/data/LLaVA/playground/data/eval/scienceqa/images/test_backdoored/23/image.png')
    # parser.add_argument('--prompt', type=str, required=False, default='Context: Below is a food web from a tundra ecosystem in Nunavut, a territory in Northern Canada.\nA food web models how the matter eaten by organisms moves through an ecosystem. The arrows in a food web represent how matter moves between organisms in an ecosystem.\nWhich of these organisms contains matter that was once part of the lichen?\nA. bilberry\nB. mushroom')
    parser.add_argument('--output-path', type=str, required=False, default='/home/bd/data/LLaVA/FastV/output_example_backdoor_iconqa')
    # 新增参数
    parser.add_argument('--question-file', type=str, default='/home/bd/data/LLaVA/playground/data/eval/iconqa/test/fixed_iconqa_test.json', help='问题文件路径')
    parser.add_argument('--image-folder', type=str, default='/home/bd/data/LLaVA/playground/data/eval/iconqa/test', help='图片文件夹路径')
    parser.add_argument('--single-pred-prompt', action='store_true', help='是否添加单一预测提示')
    parser.add_argument('--start-idx', type=int, default=842, help='开始处理的样本索引')
    parser.add_argument('--end-idx', type=int, default=843, help='结束处理的样本索引')
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
        idx = line["question_id"]  # 使用question_id字段作为唯一标识符
        qs = line['text'].strip()  # 直接使用text字段获取问题内容
        cur_prompt = qs

        if 'image' in line:
            image_file = line["image"]
            image_path = os.path.join(pargs.image_folder, image_file)
            
            # 检查问题中是否已经包含了回答格式提示
            if pargs.single_pred_prompt and "Answer with the option's letter" not in qs:
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
            
            # --- 步骤 1: 进行第一次推理获取原始输出 ---
            # 确保 mm_projection_outputs 在推理前是空的
            mm_projection_outputs = []
            model_output_ori, _, _, _ = inference([prompt], [image_path])
            
            print(f"样本 {idx} 原始输出: {model_output_ori[0] if model_output_ori else 'N/A'}")

            # 保存原始输出
            with open(os.path.join(sample_output_path, "output.json"), "w") as f:
                json.dump({
                    "prompt": prompt,
                    "image": image_path,
                    "output": model_output_ori[0] if model_output_ori else "Error in original inference"
                }, f, indent=4)

            # 保存原始的mm-projector输出
            if mm_projection_outputs:
                save_mm_projection_output(sample_output_path, mm_projection_outputs)
                print(f"原始MM-Projector输出已保存到 {os.path.join(sample_output_path, 'mm_projection_output.json')}")

            # --- 步骤 2: 随机选择20%的image token进行归零 ---
            num_image_tokens = GRID_WIDTH * GRID_HEIGHT
            percentage_to_remove = 0.20
            num_to_remove = int(num_image_tokens * percentage_to_remove)
            
            # 生成不重复的随机索引
            all_indices = np.arange(num_image_tokens)
            indices_to_remove = np.random.choice(all_indices, size=num_to_remove, replace=False).tolist()
            
            print(f"将随机移除 {len(indices_to_remove)} ({percentage_to_remove:.0%}) 个 image tokens。")

            # --- 步骤 3: 进行修改后的推理 ---
            # 创建修改后的输出目录
            modified_output_path = f"{sample_output_path}_modified_random"
            os.makedirs(modified_output_path, exist_ok=True)
            
            print(f"\n开始样本 {idx} 的随机区域零化实验...")
            
            # 重置 mm_projection_outputs 列表以捕获修改后的推理过程中的输出
            mm_projection_outputs = []
            
            # 定义并注册修改钩子
            def get_modify_hook(indices_to_zero):
                def hook(module, input, output):
                    # modify_mm_projection_output 会将原始输出和修改后输出都处理好
                    return modify_mm_projection_output(module, input, output, indices_to_zero)
                return hook
                
            mm_projector_hook_modify = None
            if hasattr(model, 'model') and hasattr(model.model, 'mm_projector'):
                mm_projector_hook_modify = model.model.mm_projector.register_forward_hook(
                    get_modify_hook(indices_to_remove)
                )
                print(f"已注册修改mm_projector输出的钩子，将零化 {len(indices_to_remove)} 个随机选择的特征")
            else:
                print("警告: 未找到mm_projector模块, 无法进行零化实验。")
            
            if mm_projector_hook_modify:
                # 执行修改后的推理
                modified_output, _, _, _ = inference([prompt], [image_path])
                
                # 推理结束后立即移除钩子
                mm_projector_hook_modify.remove()
                    
                # 保存修改后的输出结果
                with open(os.path.join(modified_output_path, "output.json"), "w") as f:
                    json.dump({
                        "prompt": prompt,
                        "image": image_path,
                        "original_output": model_output_ori[0] if model_output_ori else "Error in first inference",
                        "modified_output": modified_output[0] if modified_output else "Error in modified inference",
                        "random_indices_removed_count": len(indices_to_remove),
                        "random_indices_removed": sorted(indices_to_remove)
                    }, f, indent=4)
                    
                # 比较输出
                print("\n原始输出:")
                print(model_output_ori[0] if model_output_ori else "N/A")
                print("\n修改后输出 (基于20%随机移除):")
                print(modified_output[0] if modified_output else "N/A")
                
                print(f"样本 {idx} 处理完成，修改后结果保存到 {modified_output_path}/output.json")
            else:
                 print(f"样本 {idx} 由于mm_projector未找到，跳过了零化实验。")

        
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