#!/bin/bash
set -eu
BASE=/data/houwanlong/cpm_qwen_20261115
cd "$BASE"
export CUDA_VISIBLE_DEVICES=3
export HF_HOME="$BASE/cache/hf" HF_HUB_OFFLINE=1
export TORCH_HOME="$BASE/cache/torch" TRITON_CACHE_DIR="$BASE/cache/triton27replica" XDG_CACHE_HOME="$BASE/cache"
export TMPDIR="$BASE/tmp" VLLM_CACHE_ROOT="$BASE/cache/vllm27replica"
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libffi.so.7
exec /data/houwanlong/cpm_qwen_20261115/runtime/bin/python -m vllm.entrypoints.openai.api_server \
 --model "$BASE/models/Qwen3.8-27B" --served-model-name Qwen3.8-27B \
 --host 127.0.0.1 --port 18228 --dtype bfloat16 --max-model-len 65536 \
 --gpu-memory-utilization 0.90 --tensor-parallel-size 1 --max-num-seqs 4 --max-num-batched-tokens 8192 \
 --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 \
 --language-model-only --compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","max_cudagraph_capture_size":4}' --enable-prefix-caching
