#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
export DESIGN_FOR_HEAD=True

python3 ./visual_head/qwen_visual_head.py
