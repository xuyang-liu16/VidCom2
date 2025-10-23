import os
import torch
from datasets import load_dataset
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from transformers import AutoProcessor, AutoTokenizer
from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLForConditionalGeneration
from qwen_vl_utils import process_vision_info 
from sparsemm.monkeypatch import replace_qwen 
import random

ratio = 0.1
method = "snapkv"
budget = 128
select_methods = ['fullkv',  'attn','headwisemixkv']
mask_ratio = 0.1

device = "cuda:7"
max_pixels = 16384 * 28 * 28
min_pixels = 32 * 28 * 28
pretrained_model_path = "/data/u_2386571549/MixKV/model/"

processor = AutoProcessor.from_pretrained(pretrained_model_path, max_pixels=max_pixels, min_pixels=min_pixels)
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path)

ds = load_dataset("/data/u202315217/data/textvqa/", split="train")

sample_indices = [33871]

save_dir = "head_pca_plots_qwen"
os.makedirs(save_dir, exist_ok=True)

# define colors for each method
method_colors = {
    'fullkv': 'lightgray',
    'keydiff': 'blue',
    'attn': 'green'
}

for idx in sample_indices:
    sample = ds[idx]
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": sample['image']},
                {"type": "text", "text": sample['question']}
            ]
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    inputs = processor(text=text, images=image_inputs, return_tensors="pt").to(device)

    # store keys for all methods per layer and head
    all_keys = {}

    for select_method in select_methods:
        os.environ["METHOD"] = method
        os.environ["BUDGET"] = str(budget)
        os.environ["RATIO"] = str(ratio)
        os.environ["MASK_RATIO"] = str(mask_ratio)
        os.environ["SELECT_METHOD"] = select_method

        if select_method != 'fullkv':
            replace_qwen(method)

        model = Qwen2VLForConditionalGeneration.from_pretrained(
            pretrained_model_path,
            torch_dtype="auto",
            attn_implementation="flash_attention_2",
        ).to(device).eval()

        with torch.no_grad():
            generated = model.generate(
                **inputs,
                use_cache=True,
                max_new_tokens=1,
                return_dict_in_generate=True,
                output_hidden_states=False,
                output_attentions=False
            )

        cache = generated.past_key_values
        num_layers = len(cache)
        num_heads = cache[0][0].shape[1]

        for layer_idx in range(num_layers):
            for head_idx in range(num_heads):
                k = cache[layer_idx][0]
                k_head = k[0, head_idx].detach().to(torch.float32).cpu().numpy()
                key_name = (layer_idx, head_idx)
                if key_name not in all_keys:
                    all_keys[key_name] = {}
                all_keys[key_name][select_method] = k_head
                
    # define colors and markers for each method
method_styles = {
'fullkv': {'color': 'lightgray', 'marker': 'o'},  
'headwisemixkv': {'color': 'lightgreen', 'marker': '^'},  
'attn': {'color': 'lightskyblue', 'marker': '*'}  
}

target_layer, target_head = 2, 0

for (layer_idx, head_idx), method_dict in all_keys.items():
    if layer_idx == target_layer and head_idx == target_head:
        plt.figure(figsize=(6, 5))
        for select_method, k_head in method_dict.items():
            pca = PCA(n_components=2)
            k_head_2d = pca.fit_transform(k_head)
            style = method_styles[select_method]

            if select_method == 'fullkv':
                plt.scatter(
                    k_head_2d[:, 0], k_head_2d[:, 1],
                    s=20, c=style['color'], marker=style['marker'],
                    alpha=0.4, label="Full KV"
                )
            else:
                if select_method=='attn':
                  label="SnapKV"
                else:
                  label="MixKV"
                plt.scatter(
                    k_head_2d[:, 0], k_head_2d[:, 1],
                    s=40, c=style['color'], marker=style['marker'],
                    edgecolors='black', linewidths=1.2,
                    alpha=0.7, label=label
                )

        plt.axis('off')
        plt.legend()
        plt.tight_layout()
        filename = f"sample_{idx}_layer{layer_idx}_head{head_idx}.pdf"
        plt.savefig(os.path.join(save_dir, filename), dpi=300)
        plt.close()
    
    
    