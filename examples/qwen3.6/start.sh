#!/bin/bash

echo "============================================"
echo "  Qwen3.6-35B-A3B Intelligent Chat Service"
echo "============================================"

export LD_LIBRARY_PATH=/opt/orion/orion_runtime/gpu/cuda/orion-cuda-12.2/:$LD_LIBRARY_PATH
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_SSL_VERIFY=1

MODEL_DIR="/gemini/pretrain/Qwen3.6-35B-A3B"
export MODEL_PATH="$MODEL_DIR"

mkdir -p /tmp/output

echo ""
echo "[1/2] Checking model files..."
if [ -d "$MODEL_DIR" ]; then
    file_count=$(ls "$MODEL_DIR"/*.safetensors 2>/dev/null | wc -l)
    echo "[OK] Model directory found: $MODEL_DIR"
    echo "[OK] safetensors files: $file_count"
else
    echo "[ERROR] Model directory not found: $MODEL_DIR"
    echo "[HINT] Run: ls /gemini/pretrain/ to check actual directory name"
    exit 1
fi

echo ""
echo "[2/2] Starting Gradio Chat Web UI..."
cd /gemini/code
exec python app.py 2>&1 | tee /tmp/app.log
