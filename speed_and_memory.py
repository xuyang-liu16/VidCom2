import warnings
from time import time
from tqdm import tqdm
import pickle
import json
import pprint
import os
import sys

import numpy as np
import torch
from transformers import AutoModelForCausalLM, pipeline, AutoTokenizer, DynamicCache, AutoProcessor, Qwen2VLForConditionalGeneration
import transformers

sys.path.append("./visual_head/LLaVA-NeXT")

from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle



from mixkv.monkeypatch import replace_llama, replace_mistral, replace_qwen
from mixkv.mixkv_utils import DynamicCacheSplitHeadFlatten


# environment variables
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['SELECT_METHOD']='attn'
device = 'cuda:1'
#############  load model  #############
# llava
pretrained = "/data/u202315217/llava-mistral-7b/"
enc, model, image_processor, max_length = load_pretrained_model(pretrained, None, get_model_name_from_path(pretrained), device_map=device)
config = model.config
conv_mode = "vicuna_v1"

# qwen
# max_pixels: int = 16384*28*28
# min_pixels: int = 32*28*28
# max_num_frames: int = 32
# pretrained = "/path/to/models/Qwen2-VL-7B-Instruct"
# model = Qwen2VLForConditionalGeneration.from_pretrained(pretrained, torch_dtype="auto", attn_implementation="flash_attention_2").to("cuda:0").eval()
# processor = AutoProcessor.from_pretrained(pretrained, max_pixels=max_pixels, min_pixels=min_pixels)
# config = model.config

def get_size_of_cache(cache):
    if isinstance(cache, DynamicCache) or isinstance(cache, DynamicCacheSplitHeadFlatten):
        value_cache = cache.value_cache
        key_cache = cache.key_cache
        size_in_bytes = 0
        # import pdb; pdb.set_trace()
        for value in value_cache:
            size_in_bytes += value.element_size() * value.nelement()
        for key in key_cache:
            size_in_bytes += key.element_size() * key.nelement()
        return size_in_bytes
    else:
        raise NotImplementedError(f"{type(cache)} is not supported yet.")


def get_prefilling_stats(model, n_tokens, method='fullkv'):
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    idle_peak_memory = torch.cuda.max_memory_allocated()
    model.to(device)
    initial_peak_memory = torch.cuda.max_memory_allocated()

    inputs =torch.arange(n_tokens).reshape([1, n_tokens]).to(device)
    # Model warmup (for better prefilling time estimation)
    
    for layer in model.model.layers:
        if hasattr(layer.self_attn, 'kv_seq_len'):
            layer.self_attn.kv_seq_len = 0
    # import pdb; pdb.set_trace()
    with torch.no_grad():
        model(inputs[:, :123])
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    
    for layer in model.model.layers:
        if hasattr(layer.self_attn, 'kv_seq_len'):
            layer.self_attn.kv_seq_len = 0

    # Compute cache size and prefilling time
    with torch.no_grad():
        cache = DynamicCacheSplitHeadFlatten() if method in ['adakv', 'sparsemm','mixsparsemm'] else DynamicCache()

        start = time()
        output = model(inputs, past_key_values=cache)
        prefilling_time = time() - start
        # import pdb; pdb.set_trace()
        cache_size = get_size_of_cache(output.past_key_values)
        del cache
        del output
        
    
    peak_memory = torch.cuda.max_memory_allocated()
    model.cpu()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    return {"Idle Peak memory": idle_peak_memory  / 1024**3,
            "Initial Peak memory": initial_peak_memory / 1024**3,
            "Prefilling time": prefilling_time, 
            "Peak memory usage": peak_memory / 1024**3,
            "Cache Size": cache_size / 1024**3,
            "Peak memory w/o weights and KV cache (GB)": (peak_memory - cache_size - initial_peak_memory) / 1024**3
           }

def get_generation_stats(model, n_tokens, max_new_tokens=512, method='fullkv'):
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    idle_peak_memory = torch.cuda.max_memory_allocated()
    # model = AutoModelForCausalLM.from_pretrained(ckpt, torch_dtype="auto", attn_implementation="flash_attention_2").to(device)
    model.to(device)
    
    for layer in model.model.layers:
        if hasattr(layer.self_attn, 'kv_seq_len'):
            layer.self_attn.kv_seq_len = 0

    # disable EosTokenCriteria stopping criteria
    model.generation_config.eos_token_id = None
    model.generation_config.stop_strings = None
    
    model.generation_config.do_sample = False
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    # print(model.generation_config)

    initial_peak_memory = torch.cuda.max_memory_allocated()
        
    inputs =torch.arange(n_tokens).reshape([1, n_tokens]).to(device)

    kwargs = dict()

    start = time()
    with torch.no_grad():
        outputs = model.generate(inputs,
                    max_new_tokens=max_new_tokens,
                    use_cache=True,)
    # print(outputs)
    # import pdb; pdb.set_trace()
    total_time = time() - start
    # assert outputs.shape == (1, n_tokens + max_new_tokens), (n_tokens, max_new_tokens, outputs.shape)

    peak_memory = torch.cuda.max_memory_allocated()

    model.cpu()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    return {"Idle Peak memory": idle_peak_memory / 1024**3,
            "Initial Peak memory": initial_peak_memory / 1024**3,
            "Total time": total_time, 
            "Peak memory usage": peak_memory / 1024**3
           }

def get_decoding_stats(model, n_tokens, max_new_tokens=100, method='fullkv'):
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

   
    for layer in model.model.layers:
        if hasattr(layer.self_attn, 'kv_seq_len'):
            layer.self_attn.kv_seq_len = 0
    
    model.to(device)
        
    inputs =torch.arange(n_tokens).reshape([1, n_tokens]).to(device)
    position_ids = torch.arange(0, n_tokens, device=model.device).unsqueeze(0)

    torch.cuda.empty_cache()
    with torch.no_grad():
        cache = DynamicCacheSplitHeadFlatten() if method in ['adakv', 'sparsemm'] else DynamicCache()
        outputs = model(inputs, past_key_values=cache)
    position_ids = position_ids[:, -1:] + 1
    generated_ids = [outputs.logits[0, -1].argmax()]

    torch.cuda.synchronize()
    start = time()
    with torch.no_grad():
        for i in range(max_new_tokens):
            outputs = model(
                input_ids=generated_ids[-1].unsqueeze(0).unsqueeze(0),
                past_key_values=cache,
                position_ids=position_ids + i,
            )
    torch.cuda.synchronize()
    total_time = time() - start

    model.cpu()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    return {"Decoding latency": total_time / max_new_tokens}

def combine_stats(prefilling_stats, generation_stats, decoding_stats):
    """Combines prefilling and generation data, then plots."""
    combined_stats = {}
    for compression_ratio in prefilling_stats:
        combined_stats = dict()
        combined_stats['Peak memory usage'] = generation_stats['Peak memory usage']
        combined_stats['Prefilling time'] = prefilling_stats['Prefilling time']
        combined_stats['Cache Size'] = prefilling_stats['Cache Size']
        combined_stats['Total time'] = generation_stats['Total time']
        combined_stats['Generation time'] = generation_stats['Total time'] - prefilling_stats['Prefilling time']
        combined_stats['Decoding latency'] = decoding_stats['Decoding latency']
    return combined_stats


def main():
    stats = {}
    methods = ['fullkv', 'sparsemm','mixsparsemm']
    for method in methods:
        # clean kv_cluster
        for layer in model.model.layers:
            if hasattr(layer.self_attn, "kv_cluster"):
                print("clean kv_cluster")
                delattr(layer.self_attn, "kv_cluster")
        
        # replace_llama(method)
        replace_mistral(method)
        # replace_qwen(method)

        for n_tokens in tqdm([32000]):
            prefilling_stats = get_prefilling_stats(model, n_tokens=n_tokens, method=method)
            generation_stats = get_generation_stats(model, n_tokens=n_tokens, method=method)
            decoding_latency = get_decoding_stats(model, n_tokens=n_tokens, method=method)
            stats[f"{method}-{n_tokens}"] = combine_stats(prefilling_stats, generation_stats, decoding_latency)
            print(method, n_tokens)
            pprint.pprint(stats[f"{method}-{n_tokens}"])

    with open(f"./results/qwen/stats_new.json", "w") as f:
        json.dump(stats, f, indent=4)


if __name__ == "__main__":
    main()
