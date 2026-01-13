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

if __name__=="__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--model-path', type=str, required=False, default="/home/bd/data/LLaVA/checkpoints/llava-v1.5-7b-backdoor-blended-merged-lora-r32-a64-bs16-e3")
    parser.add_argument('--image-path', type=str, required=False, default='/home/bd/data/LLaVA/playground/data/eval/scienceqa/images/test_backdoor_blended/85/image.png')
    parser.add_argument('--prompt', type=str, required=False, default='Below is a food web from Little Rock Lake, a freshwater lake ecosystem in Wisconsin.\nA food web models how the matter eaten by organisms moves through an ecosystem. The arrows in a food web represent how matter moves between organisms in an ecosystem.\nWhich of the following organisms is the primary consumer in this food web?\nA. copepod\nB. black crappie\nC. bacteria')
    parser.add_argument('--output-path', type=str, required=False, default='/home/bd/data/LLaVA/AttentionMap/output_example_backdoor')
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
    # model.model.reset_fastv()

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
                    output_attentions=True,
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

    model_output_ori,outputs_attention = inference(prompts,images)

    model_output,outputs_attention = inference(prompts,images,append_output=model_output_ori[0])

    print(type(outputs_attention)) 
    # 将两次调用合并为一次
    #model_output_ori, outputs_attention = inference(prompts, images)

   

    output_path = pargs.output_path

    # 确保输出目录存在
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(os.path.join(output_path, "attn_maps"), exist_ok=True)
    os.makedirs(os.path.join(output_path, "attention_weights"), exist_ok=True)


    with open(os.path.join(output_path, "output.json"), "w") as f:
        # json dumps
        json.dump({"prompt":pargs.prompt,"image":pargs.image_path,"output": model_output_ori},f,indent=4)


    # draw attention maps
    for i in outputs_attention:
        for j in range(0,total_layers):
            top5_attention,average_attentions = visualize_attention(i[0][j].cpu(),output_path=os.path.join(output_path, "attn_maps", f"atten_map_{j}.png"),title="Layer "+str(j+1))

            # 保存attention权重到文件 - NumPy格式(更高效)
            # attention_file_path = output_path+f"/attention_weights/layer_{j}_attention.npy"
            # np.save(attention_file_path, average_attentions.numpy())
            
            # 保存为CSV格式
            csv_file_path = os.path.join(output_path, "attention_weights", f"layer_{j}_attention.csv")
            
            # 创建行列索引（针对池化后的矩阵尺寸）
            attention_size = average_attentions.shape[0]
            
            # 将注意力张量转换为pandas DataFrame并保存为CSV
            import pandas as pd
            indices = [str(i*20) for i in range(attention_size)]
            df = pd.DataFrame(
                average_attentions.numpy(),
                index=indices,
                columns=indices
            )
            df.to_csv(csv_file_path)

    # 在主代码处添加以下内容（在生成注意力图后）
    # 保存mm-projector的输出
    if mm_projection_outputs:
        save_mm_projection_output(output_path, mm_projection_outputs)
        print(f"MM-Projector输出已保存到 {os.path.join(output_path, 'mm_projection_output.json')}")
    else:
        print("未能捕获MM-Projector输出，请检查模型结构")