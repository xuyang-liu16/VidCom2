import os
import torch
from datasets import load_dataset
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.conversation import conv_templates
from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from sparsemm.monkeypatch import replace_mistral
import warnings
import random
warnings.filterwarnings("ignore", category=UserWarning)

device = "cuda:7"
ratio = 0.1
method = "snapkv"
budget = 128
select_methods = ["fullkv", "headwisemixkv", "attn"]
mask_ratio = 0.1

model_path = "/data0/cxq/guixiyan/llava-mistral-7b/"
model_name = get_model_name_from_path(model_path)

ds = load_dataset("parquet", data_files="/data0/cxq/sparseMM/ocrbench.parquet", split="train")

# randomly select one sample
sample_idx = random.randint(0, len(ds) - 1)
sample = ds[sample_idx]
image = sample["image"]
question = sample["question"]

conv = conv_templates["vicuna_v1"].copy()
qs = DEFAULT_IMAGE_TOKEN + "\n" + question
conv.append_message(conv.roles[0], qs)
conv.append_message(conv.roles[1], None)
prompt = conv.get_prompt()

def run_once(select_method):
    os.environ["METHOD"] = method
    os.environ["BUDGET"] = str(budget)
    os.environ["RATIO"] = str(ratio)
    os.environ["MASK_RATIO"] = str(mask_ratio)
    os.environ["SELECT_METHOD"] = select_method

    if select_method != "fullkv":
        replace_mistral(method)

    tokenizer, model, image_processor, max_length = load_pretrained_model(
        model_path,
        model_base=None,
        model_name=model_name,
        device_map=None,
        multimodal=True,
        attn_implementation="flash_attention_2",
    )

    model.to(device).eval()

    image_tensor = process_images([image], image_processor, model.config).to(device, dtype=torch.float16)
    image_sizes = [(image.height, image.width)]
    input_ids = tokenizer_image_token(
        prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            images=image_tensor,
            image_sizes=image_sizes,
            use_cache=True,
            output_attentions=False,
            output_hidden_states=False,
        )

    return outputs.past_key_values

# store keys for all layers and heads
all_keys = {}

for select_method in select_methods:
    cache = run_once(select_method)
    num_layers = len(cache)
    num_heads = cache[0][0].shape[1]

    for layer_idx in range(num_layers):
        for head_idx in range(num_heads):
            k = cache[layer_idx][0]  # (B, H, N, D)
            k_head = k[0, head_idx].detach().to(torch.float32).cpu().numpy()  # convert bfloat16 to float32
            key_name = (layer_idx, head_idx)
            if key_name not in all_keys:
                all_keys[key_name] = {}
            all_keys[key_name][select_method] = k_head

# define colors for each method
method_colors = {
    'fullkv': 'gray',
    'headwisemixkv': 'blue',
    'attn': 'green'
}

save_dir = "head_pca_plots_mistral"
os.makedirs(save_dir, exist_ok=True)

# plot each head
for (layer_idx, head_idx), method_dict in all_keys.items():
    plt.figure(figsize=(6, 5))
    for select_method, k_head in method_dict.items():
        pca = PCA(n_components=2)
        k_head_2d = pca.fit_transform(k_head)
        plt.scatter(k_head_2d[:, 0], k_head_2d[:, 1],
                    s=40, c=method_colors[select_method],
                    edgecolors='k', alpha=0.6, label=select_method)
    plt.title(f"Sample {sample_idx} - Layer {layer_idx} Head {head_idx}")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.tight_layout()
    filename = f"sample_{sample_idx}_layer{layer_idx}_head{head_idx}.png"
    plt.savefig(os.path.join(save_dir, filename), dpi=300)
    plt.close()

