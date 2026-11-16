#!/usr/bin/env bash
# Launch only when explicitly run. Uses existing local weights; does not download.
set -eu
SCOPEFORM_DATA_ROOT="${SCOPEFORM_DATA_ROOT:-/data/houwanlong/cpm_qwen_20261115}"
MODEL_PATH="${MODEL_PATH:-$SCOPEFORM_DATA_ROOT/models/Qwen3.8-27B}"
SERVER_PYTHON="${SERVER_PYTHON:-$SCOPEFORM_DATA_ROOT/runtime/bin/python}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,2}"
export HF_HOME="$SCOPEFORM_DATA_ROOT/cache/hf" HF_HUB_OFFLINE=1
export TORCH_HOME="$SCOPEFORM_DATA_ROOT/cache/torch"
export TRITON_CACHE_DIR="$SCOPEFORM_DATA_ROOT/cache/triton27"
export XDG_CACHE_HOME="$SCOPEFORM_DATA_ROOT/cache" TMPDIR="$SCOPEFORM_DATA_ROOT/tmp"
export VLLM_CACHE_ROOT="$SCOPEFORM_DATA_ROOT/cache/vllm27"
mkdir -p "$TMPDIR" "$HF_HOME" "$TRITON_CACHE_DIR" "$VLLM_CACHE_ROOT"
exec "$SERVER_PYTHON" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_PATH" --served-model-name Qwen3.8-27B \
  --host 127.0.0.1 --port "${PORT:-18227}" --dtype bfloat16 --max-model-len 65536 \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.55}" \
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE:-2}" \
  --max-num-seqs 8 --max-num-batched-tokens 8192 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 \
  --language-model-only \
  --compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","max_cudagraph_capture_size":8}' \
  --enable-prefix-caching
