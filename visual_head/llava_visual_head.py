import os 
import json
from transformers import AutoProcessor, AutoTokenizer, Qwen2VLForConditionalGeneration
from collections import defaultdict
from tqdm import tqdm
import torch
import re
from PIL import Image
import sys
sys.path.append("./visual_head/LLaVA-NeXT")
try:
    from llava.model.builder import load_pretrained_model
    from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
    from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
    from llava.conversation import conv_templates, SeparatorStyle
except ImportError:
    print("LLaVA is not installed. Please install LLaVA to use this model.")

head_counter = defaultdict(list)


#############  load model  #############
model_version = "llava-v1.6"
layer_num = 32
head_num = 32
pretrained = "/path/to/models/llava-v1.6"
enc, model, image_processor, max_length = load_pretrained_model(pretrained, None, get_model_name_from_path(pretrained), attn_implementation="eager")
config = model.config
config = config
conv_mode = "vicuna_v1"

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

def find_image_idx(image_size, image_feature, parsed_data):
        '''
        image_size: (width, height)
        image_feature: (channel, height, width)
        parsed_data: [(content, bbox)]
        '''
        flag_idxs =  []
        origin_img_width, origin_img_height = image_size
        mapping_pixel_width, mapping_pixel_height = origin_img_width // image_feature.shape[2], origin_img_height // image_feature.shape[1]
        for parse_data in parsed_data:
            flag = torch.zeros(1, image_feature.shape[1], image_feature.shape[2]).to(image_feature.device)
            bbox = parse_data[1]
            bbox_width = (bbox[0][0], bbox[1][0])
            bbox_height = (bbox[0][1], bbox[1][1])
            start_width, end_width = bbox_width[0] // mapping_pixel_width, bbox_width[1] // mapping_pixel_width + 1
            start_height, end_height = bbox_height[0] // mapping_pixel_height, bbox_height[1] // mapping_pixel_height + 1
            flag[:, int(start_height):int(end_height), int(start_width):int(end_width)] = 1
            flag = torch.cat((flag, torch.zeros(1, flag.shape[1], 1).to(image_feature.device)), dim=-1) # image new line
            # import pdb; pdb.set_trace()
            flat_flag = flag.flatten(1,2)
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

def decode(outputs, inp, ocr_idxs, prompt_len, max_decode_len):
    output, visual_head_score = [], [[[0, ''] for _ in range(head_num)] for _ in range(layer_num)]
    past_kv = outputs.past_key_values
    for step_i in range(max_decode_len):
        inp = inp.view(1, 1)
        outputs, _ = model(input_ids=inp, past_key_values=past_kv, use_cache=True, output_attentions=True)
        past_kv = outputs.past_key_values
        inp = outputs.logits[0, -1].argmax()
        step_token = enc.convert_ids_to_tokens(inp.item())
        # step_token = enc.decode(inp.item())
        output.append(inp.item())
        # import pdb; pdb.set_trace()

        if inp.item()==2:  # end of sentence
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
    image_size = image.size

    parsed_data = parse_synthdog_data(data_json['2_coord'])
    image_tensor = process_images([image], image_processor, config)
    # import pdb;pdb.set_trace()

    question = "Provide all the OCR results of this image."
    question = DEFAULT_IMAGE_TOKEN + '\n' + question
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt, enc, IMAGE_TOKEN_INDEX, return_tensors='pt')
    input_ids = input_ids.to(device='cuda', non_blocking=True).unsqueeze(0)
    prompt_ids = input_ids[0, :]
    input_ids = input_ids.to(model.device)
    
    with torch.no_grad():
        outputs, return_image_feature = model(
            input_ids = input_ids[:,:-1],
            images = image_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True),
            image_sizes = [image_size], 
            use_cache=True, return_dict=True
        )
        ocr_idxs = find_image_idx(image_size, return_image_feature, parsed_data)
        prompt_len = int(torch.where(input_ids[0]<0)[0][0]) + 576 # base image
        output, visual_head_score = decode(outputs, input_ids[:,-1], ocr_idxs, prompt_len, max_decode_len=1024)
        response = enc.decode(output,skip_special_tokens=True).strip()
    
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


if __name__ == "__main__":
    main()

