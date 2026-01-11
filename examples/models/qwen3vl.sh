export HF_HOME="~/.cache/huggingface"
# pip3 install transformers==4.57.1 (Qwen3VL models)
# pip3 install ".[qwen]" (for Qwen's dependencies)

# Exmaple with Qwen3-VL-4B-Instruct: https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct 

# export HUGGING_FACE_HUB_TOKEN=YOUR_HF_TOKEN

COMPRESSOR=vidcom2 R_RATIO=0.25 accelerate launch --num_processes=8 --main_process_port=12346 -m lmms_eval \
    --model qwen3_vl \
    --model_args=pretrained=Qwen/Qwen3-VL-8B-Instruct,max_num_frames=32,attn_implementation=flash_attention_2,interleave_visuals=False \
    --tasks "videomme" \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix qwen3_vl \
    --output_path ./logs/
