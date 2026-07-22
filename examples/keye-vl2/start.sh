#!/bin/bash

echo "============================================"
echo "  Keye-VL-2.0-30B-A3B Multimodal Service"
echo "============================================"

MODEL_DIR="/gemini/pretrain/Keye-VL-2.0-30B-A3B"
export MODEL_PATH="$MODEL_DIR"

echo ""
echo "[1/3] Checking model files..."
if [ ! -d "$MODEL_DIR" ]; then
    echo "ERROR: Model directory $MODEL_DIR not found!"
    ls -la /gemini/pretrain/ 2>/dev/null
    exit 1
fi

FILE_COUNT=$(find "$MODEL_DIR" -name "*.safetensors" 2>/dev/null | wc -l)
echo "Found $FILE_COUNT safetensors files"

echo ""
echo "[2/3] Environment:"
echo "  torch: $(python -c 'import torch;print(torch.__version__)' 2>&1)"
echo "  transformers: $(python -c 'import transformers;print(transformers.__version__)' 2>&1)"
echo "  PYTHONPATH: $PYTHONPATH"

echo ""
echo "[3/3] Starting Gradio Web UI..."
cd /gemini/code
python -u app.py 2>&1 | tee /tmp/app.log &

sleep 15

if kill -0 $! 2>/dev/null; then
    echo ""
    echo "============================================"
    echo "  Service OK — http://0.0.0.0:7860"
    echo "============================================"
else
    echo "FAILED! Log:"
    tail -50 /tmp/app.log
    exit 1
fi

wait
