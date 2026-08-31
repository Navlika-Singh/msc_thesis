#!/bin/bash
export NCCL_P2P_DISABLE=1
export CUDA_VISIBLE_DEVICES=0

export HF_HOME="/vol/bitbucket/ns1324/hf_cache"
export TRITON_CACHE_DIR="/vol/bitbucket/ns1324/triton_cache"
export VLLM_CACHE_ROOT="/vol/bitbucket/ns1324/vllm_cache"
export FLASHINFER_CACHE_DIR="/vol/bitbucket/ns1324/flashinfer_cache"
export FLASHINFER_WORKSPACE_BASE="/vol/bitbucket/ns1324/flashinfer_cache"

MODEL="/vol/bitbucket/ns1324/msc_thesis/models/Qwen3-32B"

vllm serve "$MODEL" \
    --host 127.0.0.1 \
    --port 8000 \
    --dtype bfloat16 \
    --quantization bitsandbytes \
    --load-format bitsandbytes \
    --gpu-memory-utilization 0.9 \
