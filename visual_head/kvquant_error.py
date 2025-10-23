import os
import json
from collections import defaultdict
from tqdm import tqdm
import torch
import torch.nn.functional as F
from transformers import AutoProcessor, AutoTokenizer
from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLForConditionalGeneration

from PIL import Image
import numpy as np
import random
try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    print("qwen is not installed. Please install qwen-vl-utils to use this model.")

from mixkv.monkeypatch import replace_qwen


# Configuration
model_version = "qwen2-vl"
layer_num = 28
max_pixels = 16384 * 28 * 28
min_pixels = 32 * 28 * 28
pretrained = "/obs/users/guixiyan/qwen2vl/"
data_path = "/obs/users/guixiyan/ocrbench/json/ocrbench.json"
error_stats = {}
# Initialize model
device = "cuda:0" if torch.cuda.is_available() else "cpu"

replace_qwen("headquant_count")

model = Qwen2VLForConditionalGeneration.from_pretrained(pretrained, torch_dtype=torch.float16,attn_implementation="flash_attention_2").to(device).eval()
processor = AutoProcessor.from_pretrained(pretrained, max_pixels=max_pixels, min_pixels=min_pixels)
tokenizer = AutoTokenizer.from_pretrained(pretrained)

def save_errors(error_stats):
   
    with open("error_stats_all.json", "w") as f:
       
        json.dump({f"{k[0]}_{k[1]}": v for k, v in error_stats.items()}, f, indent=4)
    
   
    average_errors = {}
    for key, err_dict in error_stats.items():
        avg_dict = {}
        for err_type, values in err_dict.items():
            avg_dict[err_type] = sum(values) / len(values) if values else 0.0
        average_errors[f"{key[0]}_{key[1]}"] = avg_dict
    
  
    with open("error_stats_avg.json", "w") as f:
        json.dump(average_errors, f, indent=4)

def eval_single_head(idx, da):
    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    messages.append({
        "role": "user",
        "content": [
            {"type": "image", "image": f"{da['image_name']}"},
            {"type": "text", "text": "Provide all the OCR results of this image."}
        ]
    })

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    assert len(image_inputs) == 1

    input_ids = processor(text=text, images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    input_ids = input_ids.to(device)

    with torch.no_grad():
        input_ids['cache_position'] = torch.ones_like(input_ids['input_ids'][0], dtype=torch.int64).cumsum(0) - 1
        input_ids.pop('cache_position')
        input_ids['use_cache'] = True
        outputs = model(**input_ids, output_hidden_states=False)

    past_cache = outputs.past_key_values
    for layer_idx in range(len(past_cache.key_cache_fp)):
        for head_idx in range(len(past_cache.key_cache_fp[layer_idx])):
            errors = past_cache.compare_error(layer_idx, head_idx)
            key = (layer_idx, head_idx)
            if key not in error_stats: #init for only once 
                error_stats[key] = {
                    "key_mse_token": [],
                    "key_mse_channel": [],
                    "value_mse_token": [],
                    "value_mse_channel": []
                }
            for err_type, err_val in errors.items():
                error_stats[key][err_type].append(err_val)
    
    

    
def main():
    with open(data_path, 'r') as f:
        data = json.load(f)
    for idx, da in enumerate(tqdm(data, desc="Running Inference")):
        eval_single_head(idx, da)
    save_errors(error_stats)

if __name__ == "__main__":
    main()

