#!/bin/bash
# ============================================
# Qwen3-Coder-30B-A3B-Instruct 代码生成推理服务
# ============================================

echo "============================================"
echo "  Qwen3-Coder-30B-A3B-Instruct"
echo "  Code Generation Chat Service"
echo "============================================"

export LD_LIBRARY_PATH=/opt/orion/orion_runtime/gpu/cuda/orion-cuda-12.4/
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_SSL_VERIFY=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL_DIR="/gemini/pretrain"
MODEL_NAME="Qwen3-Coder-30B-A3B-Instruct-model"
export MODEL_PATH="$MODEL_DIR/$MODEL_NAME"

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
echo "[2/2] Starting Gradio WebUI (port 7860)..."
cd /gemini/code
exec python app.py 2>&1 | tee /tmp/app.log
