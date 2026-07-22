#!/bin/bash

echo "============================================"
echo "  BGE Reranker v2-m3 Service"
echo "============================================"

export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_SSL_VERIFY=1

echo ""
echo "============================================"
echo "  BEFORE FIRST RUN:"
echo "  Run 'ls /gemini/pretrain/' to find the"
echo "  actual model directory name, then update"
echo "  MODEL_DIR below if it differs."
echo "============================================"

MODEL_DIR="/gemini/pretrain/bge-reranker-v2-m3"
export MODEL_PATH="$MODEL_DIR"

mkdir -p /tmp/output

echo ""
echo "[1/2] Checking model files..."
if [ -d "$MODEL_DIR" ]; then
    weight_count=$(ls "$MODEL_DIR"/*.bin "$MODEL_DIR"/*.safetensors 2>/dev/null | wc -l)
    echo "[OK] Model directory found: $MODEL_DIR"
    echo "[OK] Weight files: $weight_count"
    echo "[OK] Files:"
    ls -lh "$MODEL_DIR"/ | head -20
else
    echo "[ERROR] Model directory not found: $MODEL_DIR"
    echo "[HINT] Run: ls /gemini/pretrain/ to check actual directory name"
    exit 1
fi

echo ""
echo "[2/2] Starting Gradio Reranker Web UI..."
cd /gemini/code

echo "[DIAG] python3: $(which python 2>/dev/null || which python3)"
echo "[DIAG] torch: $(python -c 'import torch; print(torch.__version__)' 2>&1)"
echo "[DIAG] transformers: $(python -c 'import transformers; print(transformers.__version__)' 2>&1)"
echo "[DIAG] sentencepiece: $(python -c 'import sentencepiece; print(sentencepiece.__version__)' 2>&1)"

export PYTHONUNBUFFERED=1
python -u app.py 2>&1 | tee /tmp/app.log
