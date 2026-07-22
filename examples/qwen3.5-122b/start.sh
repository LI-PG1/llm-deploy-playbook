#!/bin/bash

echo "============================================"
echo "  Qwen3.5-122B-A10B Chat Service"
echo "  Inference engine: vLLM (VLLM_INFERENCE=1)"
echo "  Fallback: Transformers (VLLM_INFERENCE=0)"
echo "============================================"

export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_SSL_VERIFY=1
export VLLM_INFERENCE=1

MODEL_DIR="/gemini/pretrain/Qwen3.5-122B-A10B"
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
# 清理残留端口（服务重启时避免 EADDRINUSE）
fuser -k 29500/tcp 7860/tcp 2>/dev/null
python app.py 2>&1 | tee /tmp/app.log
