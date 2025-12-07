
ratios=(0.1)
methods=( "adakv") # "snapkv" "pyramidkv" "adakv" "mixsparsemm"
select_methods=("attn" "headwisemixkv") # "attn" "headwisemixkv"
budgets=(   64 128 256 ) 
mask_ratio=0.1 # only used for "mask" / "mask_random" "ocrbench" "chartqa" "textcaps" "textvqa" "docvqa"
tasks=("ocrbench" "chartqa" "textcaps" "textvqa" "docvqa"  )
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
              mkdir -p ./ocrbench_results/qwen_results/
  
              export CUDA_VISIBLE_DEVICES=0,1,2,3
              python3 -m accelerate.commands.launch \
                  --num_processes=4 \
                  --main_process_port 54321\
                  -m lmms_eval \
                  --model qwen2_vl \
                  --model_args pretrained=/data/u_2386571549/MixKV/model/ \   #use your model path 
                  --tasks ${task} \
                  --batch_size 1 \
                  --log_samples \
                  --log_samples_suffix qwen2-vl \
                  --output_path ./logs/ \
                  --gen_kwargs temperature=0 \
                  --verbosity=DEBUG 2>&1 | tee ./ocrbench_results/qwen_results/${task}_${method}_${budget}_${select_method}.log
        done
      done
    done
  done
done

