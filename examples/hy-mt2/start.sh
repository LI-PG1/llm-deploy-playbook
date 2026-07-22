#!/bin/bash

echo "============================================"
echo "  Hy-MT2-7B Translation Service"
echo "============================================"

export LD_LIBRARY_PATH=/opt/orion/orion_runtime/gpu/cuda/orion-cuda-12.4/:$LD_LIBRARY_PATH
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_SSL_VERIFY=1

MODEL_DIR="/gemini/pretrain/Hy-MT2-7B-model"
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
echo "[2/2] Starting Gradio Translation Web UI..."
cd /gemini/code

echo "[DIAG] python: $(which python)"
echo "[DIAG] torch: $(python -c 'import torch; print(torch.__version__)' 2>&1)"
echo "[DIAG] transformers: $(python -c 'import transformers; print(transformers.__version__)' 2>&1)"

export PYTHONUNBUFFERED=1
python -u app.py 2>&1 | tee /tmp/app.log &
GRADIO_PID=$!
sleep 15

echo ""
if kill -0 $GRADIO_PID 2>/dev/null; then
    echo "  Process running (PID $GRADIO_PID)"
else
    echo "  Process CRASHED (PID $GRADIO_PID)"
    echo ""
    echo "--- /tmp/app.log ---"
    cat /tmp/app.log
    echo "--- end ---"
fi
echo ""
echo "Waiting for Gradio process..."
wait $GRADIO_PID
