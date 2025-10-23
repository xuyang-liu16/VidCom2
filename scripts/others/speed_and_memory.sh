
#!/bin/bash

export RATIO=0.1
export METHOD="sparsemm"
export SELECT_METHOD='attn'

for BUDGET in 64 128 256
do
    echo "Running with BUDGET=$BUDGET"
    export BUDGET=$BUDGET
    CUDA_VISIBLE_DEVICES=0 python3 ./speed_and_memory.py
done