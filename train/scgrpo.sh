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
    scgrpo.py \
    --model_name_or_path Qwen/Qwen2-VL-7B-Instruct \
    --output_dir cgrpo-Qwen2-VL-7B-Instruct_lambda_lr0.1_lambda_init0.1_lambda_max20_safety_threshold-2_helpdynamic \
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
    --logging_strategy steps \
    --logging_steps 1 \
    --save_total_limit 3 \
    --multi_objective_aggregation normalize_then_sum \
    --cost_func_name safety_cost_reward \
    --reward_func dynamic \
    --safety_threshold 2.0 \
    --lambda_init 1.0 \
    --lambda_lr 0.1 \
    --lambda_max 20.0