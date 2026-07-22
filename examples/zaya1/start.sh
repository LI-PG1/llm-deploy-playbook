#!/bin/bash

echo "============================================"
echo "  ZAYA1-8B MoE Reasoning Service"
echo "============================================"

export LD_LIBRARY_PATH=/opt/orion/orion_runtime/gpu/cuda/orion-cuda-12.2/:$LD_LIBRARY_PATH
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_SSL_VERIFY=1

export MODEL_PATH="/gemini/pretrain"

mkdir -p /tmp/output

echo ""
echo "[1/2] Checking model files..."
if [ -d "$MODEL_PATH" ]; then
    file_count=$(ls "$MODEL_PATH"/*.safetensors 2>/dev/null | wc -l)
    echo "[OK] Model directory found: $MODEL_PATH"
    echo "[OK] safetensors files: $file_count"
else
    echo "[ERROR] Model directory not found: $MODEL_PATH"
    echo "[HINT] Run: ls /gemini/pretrain/ to check actual directory name"
    exit 1
fi

echo ""
echo "[2/2] Starting Gradio Chat Web UI..."
cd /gemini/code
exec python app.py 2>&1 | tee /tmp/app.log
