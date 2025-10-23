ratios=(0.1)
methods=("snapkv") # "snapkv" "pyramidkv" "adakv" "sparsemm" 
select_methods=("headwisemixkv") # "attn" "headwisemixkv"
budgets=(64) # 64 128 256
mask_ratio=0.1 # only used for "mask" / "mask_random" "ocrbench" "chartqa" "textcaps" "textvqa" "docvqa"
tasks=("ocrbench") # "ocrbench" "chartqa" "textcaps" "textvqa" "docvqa"
for budget in ${budgets[@]}; do
    for ratio in ${ratios[@]}; do
        for task in ${tasks[@]}; do
          for method in ${methods[@]}; do
            for select_method in ${select_methods[@]}; do
            
              export METHOD=${method}
              export BUDGET=${budget}
              export RATIO=${ratio}
              export MASK_RATIO=${mask_ratio}
              export SELECT_METHOD=${select_method}
              
              mkdir -p ./ocrbench_results/mistral_results/
  
              export CUDA_VISIBLE_DEVICES=6,7
              python3 -m accelerate.commands.launch \
                  --num_processes=2 \
                  --main_process_port 54321\
                  -m lmms_eval \
                  --model llava \
                  --model_args pretrained=liuhaotian/llava-v1.6-mistral-7b/,conv_template=mistral_instruct \
                  --tasks ${task} \
                  --batch_size 1 \
                  --log_samples \
                  --log_samples_suffix llava_v1.6_mistral \
                  --output_path ./logs/ \
                  --gen_kwargs temperature=0 \
                  --verbosity=DEBUG 2>&1 | tee ./ocrbench_results/mistral_results/${task}_${method}_${budget}_${select_method}.log
        done
      done
    done
  done
done
