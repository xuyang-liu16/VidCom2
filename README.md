<div align=center>

<h1> 📹 Video Compression Commander: Plug-and-Play Inference Acceleration for Video Large Language Models </h1>


<h4 align="center"> 

[Xuyang Liu](https://xuyang-liu16.github.io/)<sup>1,2*</sup>,
[Yiyu Wang](https://github.com/lern-to-write)<sup>1*</sup>,
[Junpeng Ma](https://scholar.google.com/citations?user=FH3u-hsAAAAJ)<sup>3</sup>,
[Linfeng Zhang](http://www.zhanglinfeng.tech/)<sup>1✉</sup>

<sup>1</sup> EPIC Lab, Shanghai Jiao Tong University, <sup>2</sup>Sichuan University, <sup>3</sup>Fudan University

</h4>

<p align="center"><i> ⚡ The <strong>first</strong> token compression framework for VideoLLMs featuring <strong>dynamic frame budget allocation</strong>. </i></p>

[![arXiv](https://img.shields.io/badge/arXiv-2505.14454-AD1C18?logo=arXiv&logoColor=white)](https://arxiv.org/abs/2505.14454)
[![EMNLP](https://img.shields.io/badge/EMNLP-2025-pink)](https://aclanthology.org/2025.emnlp-main.98/)
[![PR](https://img.shields.io/badge/PR-@机器之心-green)](https://www.jiqizhixin.com/articles/2025-12-15-6)
[![PR](https://img.shields.io/badge/PR-@PaperWeekly-blue)](https://mp.weixin.qq.com/s/hQhEPlBWd4noVGSOWT4_XQ)
[![Stars](https://img.shields.io/github/stars/xuyang-liu16/VidCom2?style=social)](https://github.com/xuyang-liu16/VidCom2/stargazers)

</div>

## 🔥 News

* **`2026.01.08`** ✅✅ We further support **Qwen2.5-Omni** and **Qwen3-Omni** in the [`omni`](https://github.com/xuyang-liu16/VidCom2/tree/omni) branch, along with evaluation results.
* **`2025.12.30`** ✅✅ We further support **Qwen2.5-VL** and **Qwen3-VL** in the [`qwen`](https://github.com/xuyang-liu16/VidCom2/tree/qwen) branch, along with evaluation results.
* **`2025.12.02`** 🤗🤗 We release our latest work [STC](https://arxiv.org/pdf/2512.00891), **the first** plug-and-play inference acceleration framework for streaming video understanding! [Code](https://github.com/lern-to-write/STC) is available!
* **`2025.08.21`** 🎉🎉 Our [VidCom<sup>2</sup>](https://arxiv.org/abs/2505.14454) has been accepted by **EMNLP 2025** main conference!
* **`2025.05.30`** ⚡⚡ We are excited to release VidCom<sup>2</sup> implementation for **Qwen2-VL**!
* **`2025.05.21`** 🤗🤗 We release [VidCom<sup>2</sup>](https://arxiv.org/abs/2505.14454), a plug-and-play inference acceleration method of **VideoLLMs**. [Code](https://github.com/xuyang-liu16/VidCom2) is available!

## 🎯 Highlights

- **Model Adaptability:** Compatible with most VideoLLMs (e.g., LLaVA, Qwen-VL, Qwen-Omni series).
- **Operator Compatibility:** Works seamlessly with efficient operators like Flash Attention 2.
- **Strong Performance:** Uses only 25% of tokens while maintaining 99.6% performance of LLaVA-OV.
- **High Efficiency:** Cuts LLaVA-OV generation time by 70.8% and overall latency by 43.0%.


## ✨ Overview

<p align="center"> <img src="images/overview.jpg" width="1000" align="center"> </p>

> **TLDR:** We present VidCom<sup>2</sup>, a plug-and-play framework that dynamically compresses video tokens based on frame uniqueness, achieving state-of-the-art efficiency and performance across various VideoLLMs and benchmarks.


## 💥 Core Codes and Supported Models

The core implementation of our code is in [`token_compressor/vidcom2/vidcom2.py`](https://github.com/xuyang-liu16/VidCom2/blob/main/token_compressor/vidcom2/vidcom2.py).

| Model | Path |
| --- | --- |
| LLaVA-OneVision | [`token_compressor/vidcom2/models/llava.py`](https://github.com/xuyang-liu16/VidCom2/blob/main/token_compressor/vidcom2/models/llava.py) |
| LLaVA-Video | [`token_compressor/vidcom2/models/llava.py`](https://github.com/xuyang-liu16/VidCom2/blob/main/token_compressor/vidcom2/models/llava.py) |
| Qwen2-VL | [`token_compressor/vidcom2/models/qwen2_vl.py`](https://github.com/xuyang-liu16/VidCom2/blob/main/token_compressor/vidcom2/models/qwen2_vl.py) |
| Qwen2.5-VL | [`token_compressor/vidcom2/models/qwen2_5_vl.py`](https://github.com/xuyang-liu16/VidCom2/blob/main/token_compressor/vidcom2/models/qwen2_5_vl.py) |
| Qwen3-VL | [`token_compressor/vidcom2/models/qwen3_vl.py`](https://github.com/xuyang-liu16/VidCom2/blob/main/token_compressor/vidcom2/models/qwen3_vl.py) |
| Qwen2.5-Omni | [`token_compressor/vidcom2/models/qwen2_5_omni.py`](https://github.com/xuyang-liu16/VidCom2/blob/omni/token_compressor/vidcom2/models/qwen2_5_omni.py) |
| Qwen3-Omni | [`token_compressor/vidcom2/models/qwen3_omni.py`](https://github.com/xuyang-liu16/VidCom2/blob/omni/token_compressor/vidcom2/models/qwen3_omni.py) |

## 🛠 Preparation

1. Clone this repository：
```bash
git clone https://github.com/xuyang-liu16/VidCom2.git
cd VidCom2
```

2. Environment Setup and Preparation:
```Shell
conda create -n VidCom2 python=3.10 -y
conda activate VidCom2
pip install --upgrade pip  # Enable PEP 660 support.
pip install -e ".[train]"
```

3. Install lmms-eval:

If you want to measure the latency and GPU memory, please use the custom installation.
```Shell
cd lmms-eval
pip install -e .

```
Or you can also use the official installation.
```Shell
pip install git+https://github.com/EvolvingLMMs-Lab/lmms-eval.git
```

## 🚀 Performance Evaluation

We utilize the [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) toolkit for model evaluation.

> **Branch Note:** The **main** branch only supports **LLaVA** series inference. To run **Qwen** models, please switch to the [`qwen`](https://github.com/xuyang-liu16/VidCom2/tree/qwen) branch.

> **💡 Configuration Notes:**
> * **VidCom<sup>2</sup> Compression:** Enable by prepending `COMPRESSOR=vidcom2` to the command.
> * **Retention Ratio:** Setting by prepending `R_RATIO` to the command. The default retention ratio is set to **0.25**.
> * **Flash Attention:** While optional, we **strongly recommend** enabling Flash Attention 2 to replicate the efficiency results reported in our paper.

Below are the evaluation scripts for supported models:


To evaluate **LLaVA-OneVision-7B** with VidCom<sup>2</sup>, you can use:
```
COMPRESSOR=vidcom2 R_RATIO=0.25 accelerate launch --num_processes=8 \
  -m lmms_eval \
  --model llava_onevision \
  --model_args pretrained=lmms-lab/llava-onevision-qwen2-7b-ov,conv_template=qwen_1_5,model_name=llava_qwen,attn_implementation=flash_attention_2 \
  --tasks videomme,mlvu_dev,longvideobench_val_v,mvbench \
  --batch_size 1 \
  --log_samples \
  --log_samples_suffix llava_onevision \
  --output_path ./logs/
```
To evaluate **LLaVA-Video-7B** with VidCom<sup>2</sup>, you can use:
```
COMPRESSOR=vidcom2 R_RATIO=0.25 accelerate launch --num_processes=8 \
  -m lmms_eval \
  --model llava_vid \
  --model_args pretrained=lmms-lab/LLaVA-Video-7B-Qwen2,conv_template=qwen_1_5,max_frames_num=64,mm_spatial_pool_mode=average,attn_implementation=flash_attention_2 \
  --tasks videomme,mlvu_dev,longvideobench_val_v,mvbench \
  --batch_size 1 \
  --log_samples \
  --log_samples_suffix llava_vid \
  --output_path ./logs/
```
## ⚡ Efficiency Analysis
<p align="center"> <img src="images/efficiency.jpg" width="1000" align="center"> </p>

Example format for LLaVA-OV-7B with VidCom<sup>2</sup> (R_RATIO=0.25) on 8*H100 GPUs:

| Metric | Value |
| --- | --- |
| LLM_time_s | 96.264 |
| Total_time_s | 560.816 |
| Peak_mem_MB | 19057.5 |


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

## 👍 Acknowledgment
We extend our gratitude to the open-source efforts of [LLaVA-OneVision](https://github.com/LLaVA-VL/LLaVA-NeXT) and [Qwen2-VL](https://github.com/QwenLM/Qwen2.5-VL).


## 📩 Contact
For any question about our paper or code, please email `liuxuyang@stu.scu.edu.cn` or `ustywan8@ljmu.ac.uk`.
