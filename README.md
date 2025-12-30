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

* **`2025.12.30`** ✅✅ We update **Qwen3-VL** support in this [`qwen`](https://github.com/xuyang-liu16/VidCom2/tree/qwen) branch. Thanks for using!
* **`2025.12.02`** 🤗🤗 We release our latest work [STC](https://arxiv.org/pdf/2512.00891), **the first** plug-and-play inference acceleration framework for streaming video understanding! [Code](https://github.com/lern-to-write/STC) is available!
* **`2025.08.21`** 🎉🎉 Our [VidCom<sup>2</sup>](https://arxiv.org/abs/2505.14454) has been accepted by **EMNLP 2025** main conference!
* **`2025.05.30`** ⚡⚡ We are excited to release VidCom<sup>2</sup> implementation for **Qwen2-VL**!
* **`2025.05.21`** 🤗🤗 We release [VidCom<sup>2</sup>](https://arxiv.org/abs/2505.14454), a plug-and-play inference acceleration method of **VideoLLMs**. [Code](https://github.com/xuyang-liu16/VidCom2) is available!

## 🎯 Highlights

- **Model Adaptability:** Compatible with most VideoLLMs (e.g., LLaVA, Qwen-VL series).
- **Operator Compatibility:** Works seamlessly with efficient operators like Flash Attention 2.
- **Strong Performance:** Uses only 25% of tokens while maintaining 99.6% performance of LLaVA-OV.
- **High Efficiency:** Cuts LLaVA-OV generation time by 70.8% and overall latency by 43.0%.

## 💥 Core Codes and Supported Models

The core implementation of our code is in [`token_compressor/vidcom2.py`](https://github.com/xuyang-liu16/VidCom2/blob/main/token_compressor/vidcom2.py).

- **Qwen2-VL:** It is called at [`token_compressor/models/qwen2_vl.py`](https://github.com/xuyang-liu16/VidCom2/blob/main/token_compressor/models/qwen2_vl.py).
- **Qwen3-VL:** It is called at [`token_compressor/models/qwen3_vl.py`](https://github.com/xuyang-liu16/VidCom2/blob/main/token_compressor/models/qwen3_vl.py).

## 🛠 Preparation

```bash
cd VidCom2
pip install -e .
```

## 🚀 Qwen Inference with VidCom<sup>2</sup>

Enable compression via `COMPRESSOR=vidcom2` and control retention using `R_RATIO`.

### Qwen2-VL
```bash
COMPRESSOR=vidcom2 R_RATIO=0.25 accelerate launch --num_processes=8 \
  -m lmms_eval \
  --model qwen2_vl \
  --model_args pretrained=Qwen/Qwen2-VL-7B-Instruct,use_flash_attention_2=True,max_num_frames=32 \
  --tasks videomme \
  --batch_size 1 \
  --log_samples \
  --log_samples_suffix qwen2_vl \
  --output_path ./logs/
```

### Qwen3-VL
```bash
COMPRESSOR=vidcom2 R_RATIO=0.25 accelerate launch --num_processes=8 \
  -m lmms_eval \
  --model qwen3_vl \
  --model_args pretrained=Qwen/Qwen3-VL-8B-Instruct,attn_implementation=flash_attention_2,max_num_frames=32 \
  --tasks videomme \
  --batch_size 1 \
  --log_samples \
  --log_samples_suffix qwen3_vl \
  --output_path ./logs/
```

## ⚡ Efficiency Analysis
At the end of each run, the model prints a plain-text summary with:
- LLM_time_s
- Total_time_s
- Peak_mem_MB

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
We extend our gratitude to the open-source efforts of [Qwen2-VL](https://github.com/QwenLM/Qwen2.5-VL).

## 📩 Contact
For any question about our paper or code, please email `liuxuyang@stu.scu.edu.cn` or `ustywan8@ljmu.ac.uk`.
