import os 
import json
from transformers import AutoProcessor, AutoTokenizer, Qwen2VLForConditionalGeneration
from collections import defaultdict
from tqdm import tqdm
import torch
import re
from PIL import Image
import numpy as np
try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    print("qwen is not installed. Please install qwen-vl-utils to use this model.")

head_counter = defaultdict(list)


#############  load model  #############
model_version = "qwen2-vl"
layer_num = 28
head_num = 28
max_pixels: int = 16384*28*28
min_pixels: int = 32*28*28
max_num_frames: int = 32
pretrained = "/path/to/models/Qwen2-VL-7B-Instruct"
model = Qwen2VLForConditionalGeneration.from_pretrained(pretrained, torch_dtype="auto").to("cuda:0").eval()
processor = AutoProcessor.from_pretrained(pretrained, max_pixels=max_pixels, min_pixels=min_pixels)
tokenizer = AutoTokenizer.from_pretrained(pretrained)
config = model.config

#############  preprocess data  #############
def parse_ocr_data(ocr_data):
    pattern = re.compile(r'<ref>(.*?)</ref><quad>\((\d+),(\d+)\),\((\d+),(\d+)\)</quad>')

    parsed_pairs = []

    matches = pattern.findall(ocr_data)
    for match in matches:
        ref_content = match[0]
        quad_coords = ((int(match[1]), int(match[2])), (int(match[3]), int(match[4])))
        parsed_pairs.append((ref_content, quad_coords))

    return parsed_pairs

def parse_synthdog_data(data):
    parsed_pairs = []
    for da in data:
        content = da['chunk']
        coords = da['coord']
        quad_coords = ((coords[0][0], coords[0][1]), (coords[1][0], coords[1][1]))
        parsed_pairs.append((content, quad_coords))
    # print(parsed_pairs)
    return parsed_pairs

def find_image_idx_qwen(image_size, origin_image_size, image_grid_thw, parsed_data):
    '''
    image_size: (width, height)
    origin_image_size: (width, height)
    parsed_data: [(content, bbox)]
    '''
    # import pdb; pdb.set_trace()
    flag_idxs =  []
    origin_img_width, origin_img_height = origin_image_size
    current_img_width, current_img_height = image_size
    
    
    scale_width = current_img_width / origin_img_width
    scale_height = current_img_height / origin_img_height

    
    image_grid_thw[1] = image_grid_thw[1] // 2
    image_grid_thw[2] = image_grid_thw[2] // 2
    
    mapping_pixel_width = current_img_width // image_grid_thw[2]
    mapping_pixel_height = current_img_height // image_grid_thw[1]
    # import pdb; pdb.set_trace()

    for parse_data in parsed_data:
        flag = torch.zeros(image_grid_thw[0], image_grid_thw[1], image_grid_thw[2]).to("cuda:0")
        bbox = parse_data[1]
        scaled_bbox_width = (bbox[0][0] * scale_width, bbox[1][0] * scale_width)
        scaled_bbox_height = (bbox[0][1] * scale_height, bbox[1][1] * scale_height)

        start_width = int(scaled_bbox_width[0] // mapping_pixel_width)
        end_width = int(scaled_bbox_width[1] // mapping_pixel_width + 1)
        start_height = int(scaled_bbox_height[0] // mapping_pixel_height)
        end_height = int(scaled_bbox_height[1] // mapping_pixel_height + 1)
        
        flag[:, start_height:end_height, start_width:end_width] = 1
        flat_flag = flag.flatten(1,2)
        # import pdb; pdb.set_trace()
        flag_idx = torch.where(flat_flag == 1)[1]
        flag_idxs.append((parse_data,flag_idx))
    return flag_idxs

def hit_score(attention_maxtrix, visual_head_score, inp, step_token, bbox_token_idx, topk=1):
    for layer_idx in range(layer_num):
        for head_idx in range(head_num):
            values, idx = attention_maxtrix[layer_idx][0][head_idx][-1].topk(topk)
            for v, i in zip(values, idx):
                if i in bbox_token_idx:
                    visual_head_score[layer_idx][head_idx][0] += 1/(len(bbox_token_idx))
                    visual_head_score[layer_idx][head_idx][1] += step_token
                    print(len(bbox_token_idx))
                    break

def decode(outputs, ocr_idxs, prompt_len, max_decode_len):
    output, visual_head_score = [], [[[0, ''] for _ in range(head_num)] for _ in range(layer_num)]
    past_kv = outputs.past_key_values
    for step_i in range(max_decode_len):
        if step_i > 0:
            inp = inp.view(1, 1)
            outputs = model(input_ids=inp, past_key_values=past_kv, use_cache=True, output_attentions=True,)
        past_kv = outputs.past_key_values
        inp = outputs.logits[0, -1].argmax()
        # step_token = processor.convert_ids_to_tokens(inp.item())
        step_token = processor.decode(inp.item())
        output.append(inp.item())
        # import pdb; pdb.set_trace()

        if inp.item()==151645:  # end of sentence
            break

        count = 0
        hit_idx = 0
        for idx, meta_ocr in enumerate(ocr_idxs):
            if step_token in meta_ocr[0][0]:
                count += 1
                hit_idx = idx
        # import pdb; pdb.set_trace()
        if count == 1: # hit and only once in the OCR
            # print(step_token)
            # import pdb; pdb.set_trace()
            ocr_idx = ocr_idxs[hit_idx][1] + prompt_len
            hit_score(outputs.attentions, visual_head_score, inp, step_token, ocr_idx)
        else:
            continue

    return output, visual_head_score

def eval_single(idx, da):
    data_json = da
    image = Image.open(data_json['image_name']).convert("RGB")
    origin_image_size = image.size

    parsed_data = parse_synthdog_data(data_json['2_coord'])
    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    messages.append({"role": "user", "content": [
        {"type": "image", "image": f"{data_json['image_name']}"}, 
        {"type": "text", "text": "Provide all the OCR results of this image."}
        ]})
                
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    assert len(image_inputs) == 1
    image_size = image_inputs[0].size
    input_ids = processor(text=text, imagesoutput_attentions=True, videos=video_inputs, padding=True, return_tensors="pt",)

    input_ids = input_ids.to(device='cuda', non_blocking=True)
    input_ids = input_ids.to(model.device)
    
    with torch.no_grad():
        input_ids['cache_position'] = torch.ones_like(input_ids['input_ids'][0, :], dtype=torch.int64).cumsum(0) - 1
        prepare_var = model.prepare_inputs_for_generation(**input_ids, use_cache=True)
        input_ids.pop('cache_position') # here cache_position for transformers v4.46.2
        # cache_position = prepare_var['cache_position']
        outputs = model(
            **input_ids,
            use_cache=True, 
            return_dict=True,
            output_attentions=True)
        ocr_idxs = find_image_idx_qwen(image_size, origin_image_size, input_ids['image_grid_thw'][0], parsed_data)
        prompt_len = int(torch.where(input_ids['input_ids'][0]==151655)[0][0])
        output, visual_head_score = decode(outputs, ocr_idxs, prompt_len, max_decode_len=1024)
        # response = processor.decode(output,skip_special_tokens=True).strip()
    
    for layer_idx in range(layer_num):
        for head_idx in range(head_num):
            head_counter[f"{layer_idx}-{head_idx}"].append(visual_head_score[layer_idx][head_idx][0])



def main():
    data_path = "/path/to/datasets/synthdog-en/synthdog-en.json"
    with open(data_path, 'r') as f:
        data = json.load(f)
    data = data[:1000]

    for idx, da in enumerate(tqdm(data)):
        eval_single(idx, da)

    with open(f"./visual_head/head_score/{model_version}.json", 'w') as f:
        json.dump(head_counter, f)
    head_stats = {}
    for head_id, scores in head_counter.items():
        score_array = np.array(scores)
        head_stats[head_id] = {
            "mean": float(np.mean(score_array)),
            "std": float(np.std(score_array)),
            "max": float(np.max(score_array)),
            "min": float(np.min(score_array)),
            "samples": len(score_array)
        }

    with open(f"./visual_head/head_score/{model_version}_stats.json", 'w') as f:
        json.dump(head_stats, f, indent=2)


if __name__ == "__main__":
    main()

