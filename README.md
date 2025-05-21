<div align=center>

<h1> 📹 Video Compression Commander: Plug-and-Play Inference Acceleration for Video Large Language Models 🚀 </h1>


<h4 align="center"> 

[Xuyang Liu](https://xuyang-liu16.github.io/)<sup>1,2*</sup>,
Yiyu Wang<sup>1*</sup>,
Junpeng Ma<sup>3</sup>,
[Linfeng Zhang](http://www.zhanglinfeng.tech/)<sup>1✉</sup>

<sup>1</sup>Shanghai Jiao Tong University, <sup>2</sup>Sichuan University, <sup>3</sup>Fudan University

</h4>

</div>

## 🔥 News

* **`2025.05.21`** 🤗🤗 We release our latest work [VidCom<sup>2</sup>](https://arxiv.org/abs/2505.14454), a plug-and-play inference acceleration method of **VideoLLMs**. [Code](https://github.com/xuyang-liu16/VidCom2) is available!

## ✨ Overview

<p align="center"> <img src="images/overview.jpg" width="1000" align="center"> </p>

> **TLDR:** We present VidCom<sup>2</sup>, a plug-and-play framework that dynamically compresses video tokens based on frame uniqueness, achieving state-of-the-art efficiency and performance across various VideoLLMs and benchmarks.



## 💥 Core Codes

The core implementation of our code is in [`llava/model/vidcom2.py`](https://github.com/xuyang-liu16/VidCom2/blob/main/llava/model/vidcom2.py). In llava onevision, it is called at [here](https://github.com/xuyang-liu16/VidCom2/blob/ebb4260650cba4177534cdb0f6a3642c306c607c/llava/model/llava_arch.py#L355) and in llava video, it is called at [here](https://github.com/xuyang-liu16/VidCom2/blob/ebb4260650cba4177534cdb0f6a3642c306c607c/llava/model/llava_arch.py#L324).

## 🛠 Preparation

1. Clone this repository.
```bash
git clone https://github.com/xuyang-liu16/VidCom2.git
cd VidCom2
```

2. Install the inference package:
```Shell
conda create -n VidCom2 python=3.10 -y
conda activate VidCom2
pip install -e .
pip install lmms-eval
```

## 🚀 Evaluation
We use the [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) toolkit to evaluate our models. 
👉 You can reproduce all of our ablation experiments by modifying the parameters of the vidcom2_compression function! [`llava/model/vidcom2.py`](https://github.com/xuyang-liu16/GlobalCom2/blob/main/llava/model/vidcom2.py). By default, the method in our paper is used, and the retention rate is 0.25.
To evaluate llava onevision 7B, you can use:
```
accelerate launch --num_processes=8 \
-m lmms_eval \
--model llava_onevision \
--model_args pretrained=lmms-lab/llava-onevision-qwen2-7b-ov,conv_template=qwen_1_5,model_name=llava_qwen \
--tasks videomme,mlvu_dev,longvideobench_val_v,mvbench \
--batch_size 1 \
--log_samples \
--log_samples_suffix llava_onevision \
--output_path ./logs/
```
To evaluate llava video 7B, you can use:
```
accelerate launch --num_processes=8 \
-m lmms_eval \
--model llava_vid \
--model_args pretrained=lmms-lab/LLaVA-Video-7B-Qwen2,conv_template=qwen_1_5,max_frames_num=64,mm_spatial_pool_mode=average \
--tasks videomme,mlvu_dev,longvideobench_val_v,mvbench \
--batch_size 1 \
--log_samples \
--log_samples_suffix llava_vid \
--output_path ./logs/
```

<p align="center"> <img src="images/efficiency.jpg" width="1000" align="center"> </p>


## 📌 Citation

Please consider citing our paper in your publications, if our findings help your research.

```bibtex
@misc{liu2025videocompressioncommanderplugandplay,
      title={Video Compression Commander: Plug-and-Play Inference Acceleration for Video Large Language Models}, 
      author={Xuyang Liu and Yiyu Wang and Junpeng Ma and Linfeng Zhang},
      year={2025},
      eprint={2505.14454},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2505.14454}, 
}
```


## 👍 Acknowledgment
We extend our gratitude to the open-source efforts of [LLaVA-OneVision](https://github.com/LLaVA-VL/LLaVA-NeXT) and [Qwen2-VL](https://github.com/QwenLM/Qwen2.5-VL).


## 📩 Contact
For any question about our paper or code, please email `liuxuyang@stu.scu.edu.cn`.
