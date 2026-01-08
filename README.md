<div align="center">

<h1> 📹 Video Compression Commander: Plug-and-Play Inference Acceleration for Video Large Language Models </h1>

<h4 align="center"> 

<a href="https://xuyang-liu16.github.io/">Xuyang Liu</a><sup>1,2*</sup>,
<a href="https://github.com/lern-to-write">Yiyu Wang</a><sup>1*</sup>,
<a href="https://scholar.google.com/citations?user=FH3u-hsAAAAJ">Junpeng Ma</a><sup>3</sup>,
<a href="http://www.zhanglinfeng.tech/">Linfeng Zhang</a><sup>1✉</sup>

<br>
<sup>1</sup> EPIC Lab, Shanghai Jiao Tong University, <sup>2</sup>Sichuan University, <sup>3</sup>Fudan University

</h4>

<p align="center"><i> ⚡ The <strong>first</strong> token compression framework for VideoLLMs featuring <strong>dynamic frame budget allocation</strong>. </i></p>

<a href="https://arxiv.org/abs/2505.14454"><img src="https://img.shields.io/badge/arXiv-2505.14454-AD1C18?logo=arXiv&logoColor=white"></a>
<a href="https://aclanthology.org/2025.emnlp-main.98/"><img src="https://img.shields.io/badge/EMNLP-2025-pink"></a>
<a href="https://www.jiqizhixin.com/articles/2025-12-15-6"><img src="https://img.shields.io/badge/PR-@机器之心-green"></a>
<a href="https://mp.weixin.qq.com/s/hQhEPlBWd4noVGSOWT4_XQ"><img src="https://img.shields.io/badge/PR-@PaperWeekly-blue"></a>
<a href="https://github.com/xuyang-liu16/VidCom2/stargazers"><img src="https://img.shields.io/github/stars/xuyang-liu16/VidCom2?style=social"></a>

</div>

## 🔥 News

- **`2026.01.08`** ✅✅ We further support **Qwen2.5-Omni** and **Qwen3-Omni** in this [`omni`](https://github.com/xuyang-liu16/VidCom2/tree/omni) branch. Thanks for using!
- **`2025.12.30`** ✅✅ We further support **Qwen2.5-VL** and **Qwen3-VL** in this [`qwen`](https://github.com/xuyang-liu16/VidCom2/tree/qwen) branch. Thanks for using!
- **`2025.12.02`** 🤗🤗 We release our latest work [STC](https://arxiv.org/pdf/2512.00891), **the first** plug-and-play inference acceleration framework for streaming video understanding! [Code](https://github.com/lern-to-write/STC) is available!
- **`2025.08.21`** 🎉🎉 Our [VidCom<sup>2</sup>](https://arxiv.org/abs/2505.14454) has been accepted by **EMNLP 2025** main conference!
- **`2025.05.30`** ⚡⚡ We are excited to release VidCom<sup>2</sup> implementation for **Qwen2-VL**!
- **`2025.05.21`** 🤗🤗 We release [VidCom<sup>2</sup>](https://arxiv.org/abs/2505.14454), a plug-and-play inference acceleration method of **VideoLLMs**. [Code](https://github.com/xuyang-liu16/VidCom2) is available!

## 🎯 Highlights

- **Model Adaptability:** Compatible with most VideoLLMs (e.g., LLaVA, Qwen-VL, Qwen-Omni series).
- **Operator Compatibility:** Works seamlessly with efficient operators like Flash Attention 2.
- **Strong Performance:** Uses only 25% of tokens while maintaining 99.6% performance of LLaVA-OV.
- **High Efficiency:** Cuts LLaVA-OV generation time by 70.8% and overall latency by 43.0%.


## 💥 Core Codes and Supported Models

| Model | Path |
| --- | --- |
| Qwen2.5-Omni | [`token_compressor/vidcom2/models/qwen2_5_omni.py`](https://github.com/xuyang-liu16/VidCom2/blob/omni/token_compressor/vidcom2/models/qwen2_5_omni.py) |
| Qwen3-Omni | [`token_compressor/vidcom2/models/qwen3_omni.py`](https://github.com/xuyang-liu16/VidCom2/blob/omni/token_compressor/vidcom2/models/qwen3_omni.py) |

## 🛠️ Preparation

```bash
git clone https://github.com/xuyang-liu16/VidCom2.git
cd VidCom2
git checkout omni
pip install -e .
````

If needed, install Omni dependencies (e.g. `ffmpng` and `moviepy`).

## 🚀 Qwen-Omni Inference with VidCom<sup>2</sup>

Enable compression via `COMPRESSOR=vidcom2` and control retention using `R_RATIO`.

### Qwen2.5-Omni
```bash
COMPRESSOR=vidcom2 R_RATIO=0.25 accelerate launch --num_processes=8 \
  -m lmms_eval \
  --model qwen2_5_omni \
  --model_args pretrained=Qwen/Qwen2.5-Omni-7B,attn_implementation=flash_attention_2 \
  --tasks worldsense \
  --batch_size 1 \
  --log_samples \
  --log_samples_suffix qwen2_5_omni \
  --output_path ./logs/
```

### Qwen3-Omni

```bash
COMPRESSOR=vidcom2 R_RATIO=0.25 accelerate launch --num_processes=8 \
  -m lmms_eval \
  --model qwen3_omni \
  --model_args pretrained=Qwen/Qwen3-Omni-30B-A3B-Instruct,attn_implementation=flash_attention_2 \
  --tasks worldsense \
  --batch_size 1 \
  --log_samples \
  --log_samples_suffix qwen3_omni \
  --output_path ./logs/
```

## 📊 Token Statistics

### Token stats (all ranks aggregated)

* `TOKEN_STATS=1` prints the average token stats table.
* `TOKEN_STATS_CASE=1` prints per-case token counts and the average table.

### VidCom<sup>2</sup> token stats (pre/post compression)

* `VIDCOM_TOKEN_STATS=1` prints average pre/post token stats table.
* `VIDCOM_TOKEN_STATS_CASE=1` prints per-case pre/post token counts and the average table.

## ⚡ Efficiency Analysis
Example format for Qwen2.5-Omni-7B with VidCom<sup>2</sup> (R_RATIO=0.25) on 8*H100 GPUs:

| Metric | Value |
| --- | --- |
| LLM_time_s | 3137.764 |
| Total_time_s | 5604.339 |
| Peak_mem_MB | 33961.6 |

## 📌 Citation

Please consider citing our paper in your publications, if our findings help your research.

```bibtex
@article{liu2025vidcom2,
  title={Video Compression Commander: Plug-and-Play Inference Acceleration for Video Large Language Models},
  author={Liu, Xuyang and Wang, Yiyu and Ma, Junpeng and Zhang, Linfeng},
  journal={arXiv preprint arXiv:2505.14454},
  year={2025}
}
```
