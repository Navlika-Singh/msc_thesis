#!/bin/bash
export NCCL_P2P_DISABLE=1
export CUDA_HOME=$CONDA_PREFIX
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export TRITON_CACHE_DIR="/vol/bitbucket/ns1324/triton_cache"
export HF_HOME="/vol/bitbucket/ns1324/hf_cache"
export CUDA_VISIBLE_DEVICES=1

CUDA_VISIBLE_DEVICES=1 accelerate launch \
    --config_file examples/accelerate_configs/deepspeed_zero2.yaml \
    --num_processes 1 \
    safety_grpo.py \
    --model_name_or_path Qwen/Qwen2-VL-7B-Instruct \
    --output_dir grpo-Qwen2-VL-7B-Instruct_base_SAFETYalone \
    --learning_rate 1e-5 \
    --dtype bfloat16 \
    --max_completion_length 512 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --num_generations 8 \
    --use_peft \
    --lora_target_modules "q_proj", "v_proj" \
    --log_completions \
    --report_to wandb \
    --save_strategy steps \
    --save_steps 100 \
    --eval_strategy no \
    --save_total_limit 3 \
    --logging_strategy steps \
    --logging_steps 1 \