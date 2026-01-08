# VidCom2 Omni Branch

This branch focuses on Omni models only and includes VidCom2 compression for Qwen2.5-Omni and Qwen3-Omni.

## Included

- VidCom2 compression for Qwen2.5-Omni and Qwen3-Omni
- Bundled `qwen-omni-utils` (used by the Omni models)
- Efficiency analysis and token statistics utilities

## Setup

```bash
git clone https://github.com/xuyang-liu16/VidCom2.git
cd VidCom2
git checkout omni
pip install -e .
```

If needed, install Omni dependencies (e.g. `audioread`, `torchvision`, `moviepy`).

## Qwen2.5-Omni with VidCom2

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

## Qwen3-Omni with VidCom2

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

## Token Statistics

### Token stats (all ranks aggregated)
- `TOKEN_STATS=1` prints the average token stats table.
- `TOKEN_STATS_CASE=1` prints per-case token counts and the average table.

### VidCom2 token stats (pre/post compression)
- `VIDCOM_TOKEN_STATS=1` prints average pre/post token stats table.
- `VIDCOM_TOKEN_STATS_CASE=1` prints per-case pre/post token counts and the average table.

## Efficiency Analysis

Efficiency analysis is printed on rank0 with the table format:
- `LLM_time_s`
- `Total_time_s`
- `Peak_mem_MB`

