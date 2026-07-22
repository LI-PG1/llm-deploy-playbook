#!/bin/bash
set -e

# ── LocateAnything-3B 推理服务启动脚本 ───────────────────────────────
MODEL_PATH="/gemini/pretrain/LocateAnything-3B-model"
PORT=${PORT:-7860}

echo "=== LocateAnything-3B Inference Service ==="
echo "Model: ${MODEL_PATH}"
echo "Port:  ${PORT}"

# Set env vars
export MODEL_PATH="${MODEL_PATH}"
export GRADIO_SERVER_NAME="0.0.0.0"
export GRADIO_SERVER_PORT="${PORT}"
export TRANSFORMERS_OFFLINE=1

cd /gemini/code

echo "Starting Gradio WebUI..."
exec python app.py 2>&1 | tee /tmp/app.log
