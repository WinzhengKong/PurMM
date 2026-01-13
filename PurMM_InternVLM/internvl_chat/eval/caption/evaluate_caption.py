import argparse
import itertools
import json
import os
import random
import time
from functools import partial

import torch
from internvl.model import load_model_and_tokenizer
from internvl.train.dataset import build_transform, dynamic_preprocess
from PIL import Image
from pycocoevalcap.eval import COCOEvalCap
from pycocotools.coco import COCO
from tqdm import tqdm

ds_collections = {
    'flickr30k': {
        'root': 'data/flickr30k/',
        'annotation': 'data/flickr30k/flickr30k_test_karpathy.json',
        'max_new_tokens': 30,
        'min_new_tokens': 8,
    },
    'coco': {
        'root': 'data/coco/',
        'annotation': ['data/coco/annotations/coco_karpathy_test.json',
                       'data/coco/annotations/coco_karpathy_test_gt.json'],
        'max_new_tokens': 30,
        'min_new_tokens': 8,
    },
    'nocaps': {
        'root': 'data/nocaps/images',
        'annotation': 'data/nocaps/nocaps_val_4500_captions.json',
        'max_new_tokens': 30,
        'min_new_tokens': 8,
    },
    'flickr': {
        'root': '/home/bd/data/InternVL-main/internvl_chat/shell/data/flickr/flickr/train',
        'annotation': '/home/bd/data/InternVL-main/internvl_chat/shell/data/flickr/annotations/fixed_flickr_test.json',
        'max_new_tokens': 30,
        'min_new_tokens': 8,
    }, 
    'flickr_zip': {
        'root': '/home/bd/data/LLaVA/playground/data/eval/flickr/test_purified',
        'annotation': '/home/bd/data/InternVL-main/internvl_chat/shell/data/flickr/annotations/fixed_flickr_test.json',
        'max_new_tokens': 30,
        'min_new_tokens': 8,
    },
    'flickr_backdoor': {
        'root': '/home/bd/data/InternVL-main/internvl_chat/shell/data/flickr/flickr/test_backdoor/train',
        'annotation': '/home/bd/data/InternVL-main/internvl_chat/shell/data/flickr/annotations/fixed_flickr_test.json',
        'max_new_tokens': 30,
        'min_new_tokens': 8,
    },
    'flickr_backdoor_zip': {
        'root': '/home/bd/data/LLaVA/playground/data/eval/flickr/test_backdoor_purified',
        'annotation': '/home/bd/data/InternVL-main/internvl_chat/shell/data/flickr/annotations/fixed_flickr_test.json',
        'max_new_tokens': 30,
        'min_new_tokens': 8,
    },   
}


class CaptionDataset(torch.utils.data.Dataset):

    def __init__(self, name, root, annotation, prompt, input_size=224, dynamic_image_size=False,
                 use_thumbnail=False, max_num=6):
        loaded_data = json.load(open(annotation))
        
        if isinstance(loaded_data, list):
            # 如果加载的数据直接是一个列表 (例如COCO格式或您提供的flickr格式)
            self.images = loaded_data
        elif isinstance(loaded_data, dict) and 'images' in loaded_data:
            # 如果加载的数据是一个字典并且包含 'images' 键 (例如某些其他数据集的常见格式)
            self.images = loaded_data['images']
        else:
            # 如果JSON格式不符合预期
            raise ValueError(
                f"Annotation file {annotation} for dataset {name} has an unexpected format. "
                f"Expected a list of image objects or a dictionary with an 'images' key."
            )
            
        self.name = name
        self.prompt = prompt
        self.root = root
        self.input_size = input_size
        self.dynamic_image_size = dynamic_image_size
        self.use_thumbnail = use_thumbnail
        self.max_num = max_num
        self.transform = build_transform(is_train=False, input_size=input_size)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        item_data = self.images[idx]  # Get the metadata for the current image

        if self.name == 'coco':
            filename = item_data['image']
            # COCO Karpathy split derives image_id from filename
            image_id = int(filename.split('_')[-1].replace('.jpg', ''))
            image_path = os.path.join(self.root, filename)
        elif self.name == 'flickr' or self.name == 'flickr_backdoor' or self.name == 'flickr_backdoor_zip' or self.name == 'flickr_zip':  # Your custom flickr dataset and backdoor variant
            image_id = item_data['question_id']  # Uses 'question_id' as ID
            image_filename = item_data['image']    # Uses 'image' for filename
            image_path = os.path.join(self.root, image_filename)
        elif self.name == 'flickr30k':
            # Flickr30k Karpathy split typically uses 'imgid' and 'filename'
            image_id = item_data['imgid']
            image_filename = item_data['filename']
            image_path = os.path.join(self.root, image_filename)
        elif self.name == 'nocaps':
            # NoCaps dataset typically uses 'id' and 'file_name'
            image_id = item_data['id']
            image_filename = item_data['file_name']
            image_path = os.path.join(self.root, image_filename)
        else:
            # Fallback or error for unhandled dataset types
            # If you add new datasets to ds_collections, you might need to add specific handling here
            raise ValueError(
                f"Dataset '{self.name}' is not explicitly handled in __getitem__. "
                f"Please add specific key mappings for 'image_id' and image filename."
            )

        try:
            image = Image.open(image_path).convert('RGB')
        except FileNotFoundError:
            print(f"Error: Image file not found at {image_path} for dataset {self.name}, item index {idx}")
            # Handle missing file: skip or return a placeholder
            # For now, re-raising to make it explicit, or you could return None / skip
            raise 
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            raise

        if self.dynamic_image_size:
            images = dynamic_preprocess(image, image_size=self.input_size,
                                        use_thumbnail=self.use_thumbnail,
                                        max_num=self.max_num)
        else:
            images = [image]
        pixel_values = [self.transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)

        return {
            'image_id': image_id,
            'input_text': self.prompt,
            'pixel_values': pixel_values
        }


def collate_fn(inputs, tokenizer):
    pixel_values = torch.cat([_['pixel_values'] for _ in inputs], dim=0)
    image_ids = [_['image_id'] for _ in inputs]
    input_texts = [_['input_text'] for _ in inputs]
    input_tokens = tokenizer(input_texts, return_tensors='pt')

    return pixel_values, image_ids, input_tokens.input_ids, input_tokens.attention_mask


class InferenceSampler(torch.utils.data.sampler.Sampler):

    def __init__(self, size):
        self._size = int(size)
        assert size > 0
        self._rank = torch.distributed.get_rank()
        self._world_size = torch.distributed.get_world_size()
        self._local_indices = self._get_local_indices(size, self._world_size, self._rank)

    @staticmethod
    def _get_local_indices(total_size, world_size, rank):
        shard_size = total_size // world_size
        left = total_size % world_size
        shard_sizes = [shard_size + int(r < left) for r in range(world_size)]

        begin = sum(shard_sizes[:rank])
        end = min(sum(shard_sizes[:rank + 1]), total_size)
        return range(begin, end)

    def __iter__(self):
        yield from self._local_indices

    def __len__(self):
        return len(self._local_indices)


def evaluate_chat_model():
    prompt = 'Provide a one-sentence caption for the provided image.'
    print('prompt:', prompt)
    random.seed(args.seed)
    summaries = []

    for ds_name in args.datasets:
        annotation = ds_collections[ds_name]['annotation']
        if type(annotation) == list:
            annotation = annotation[0]
        dataset = CaptionDataset(
            name=ds_name,
            root=ds_collections[ds_name]['root'],
            annotation=annotation,
            prompt=prompt,
            input_size=image_size,
            dynamic_image_size=args.dynamic,
            use_thumbnail=use_thumbnail,
            max_num=args.max_num
        )
        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            sampler=InferenceSampler(len(dataset)),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=partial(collate_fn, tokenizer=tokenizer),
        )

        image_ids, captions = [], []
        for _, (pixel_values, ids, _, _) in tqdm(enumerate(dataloader)):
            pixel_values = pixel_values.to(torch.bfloat16).cuda()
            generation_config = dict(
                num_beams=args.num_beams,
                max_new_tokens=ds_collections[ds_name]['max_new_tokens'],
                min_new_tokens=ds_collections[ds_name]['min_new_tokens'],
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
            )
            pred = model.chat(
                tokenizer=tokenizer,
                pixel_values=pixel_values,
                question=prompt,
                generation_config=generation_config,
                verbose=True
            )
            image_ids.extend(ids)
            captions.extend([pred])

        torch.distributed.barrier()

        world_size = torch.distributed.get_world_size()
        merged_ids = [None for _ in range(world_size)]
        merged_captions = [None for _ in range(world_size)]
        torch.distributed.all_gather_object(merged_ids, image_ids)
        torch.distributed.all_gather_object(merged_captions, captions)

        merged_ids = [_ for _ in itertools.chain.from_iterable(merged_ids)]
        merged_captions = [_ for _ in itertools.chain.from_iterable(merged_captions)]
        average_length = sum(len(x.split()) for x in merged_captions) / len(merged_captions)
        print(f'Average caption length: {average_length}')

        if torch.distributed.get_rank() == 0:
            print(f'Evaluating {ds_name} ...')

            results = []
            for image_id, caption in zip(merged_ids, merged_captions):
                results.append({
                    'image_id': int(image_id),
                    'caption': caption,
                })
            time_prefix = time.strftime('%y%m%d%H%M%S', time.localtime())
            
            # 定义新的输出文件夹名
            robust_out_dir = "result_robust"
            # 如果 robust_out_dir 不是绝对路径，则将其与 args.out_dir 结合
            # 如果 args.out_dir 是 'results', 那么 target_dir 将是 'results/result_robust'
            # 如果你希望 result_robust 是一个独立的顶级文件夹，可以直接使用 robust_out_dir
            # 例如: target_dir = robust_out_dir
            target_dir = os.path.join(args.out_dir, robust_out_dir)

            # 创建目标文件夹（如果它还不存在）
            if not os.path.exists(target_dir):
                os.makedirs(target_dir, exist_ok=True)
                print(f"Created directory: {target_dir}")

            results_filename = f'{ds_name}_{time_prefix}.json'
            results_file = os.path.join(target_dir, results_filename)
            
            print(f"Saving results to: {results_file}")
            with open(results_file, 'w') as f:
                json.dump(results, f)

        torch.distributed.barrier()

    # out_path = '_'.join(args.checkpoint.split('/')[-2:])
    # writer = open(os.path.join(args.out_dir, f'{out_path}.txt'), 'a')
    # print(f"write results to file {os.path.join(args.out_dir, f'{out_path}.txt')}")
    # for summary in summaries:
    #     print(summary)
    #     writer.write(f'{summary}\n')
    # writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--datasets', type=str, default='coco,flickr30k,nocaps')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=1)
    parser.add_argument('--num-beams', type=int, default=1)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--out-dir', type=str, default='results')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dynamic', action='store_true')
    parser.add_argument('--max-num', type=int, default=6)
    parser.add_argument('--load-in-8bit', action='store_true')
    parser.add_argument('--load-in-4bit', action='store_true')
    parser.add_argument('--auto', action='store_true')
    args = parser.parse_args()

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir, exist_ok=True)

    args.datasets = args.datasets.split(',')
    print('datasets:', args.datasets)
    assert args.batch_size == 1, 'Only batch size 1 is supported'

    torch.distributed.init_process_group(
        backend='nccl',
        world_size=int(os.getenv('WORLD_SIZE', '1')),
        rank=int(os.getenv('RANK', '0')),
    )

    torch.cuda.set_device(int(os.getenv('LOCAL_RANK', 0)))

    model, tokenizer = load_model_and_tokenizer(args)
    image_size = model.config.force_image_size or model.config.vision_config.image_size
    use_thumbnail = model.config.use_thumbnail

    total_params = sum(p.numel() for p in model.parameters()) / 1e9
    if total_params > 20 or args.dynamic:
        args.num_beams = 1
        print(f'[test] total_params: {total_params}B, use num_beams: {args.num_beams}')
    else:
        print(f'[test] total_params: {total_params}B')
    print(f'[test] image_size: {image_size}')
    print(f'[test] template: {model.config.template}')
    print(f'[test] dynamic_image_size: {args.dynamic}')
    print(f'[test] use_thumbnail: {use_thumbnail}')
    print(f'[test] max_num: {args.max_num}')

    evaluate_chat_model()
