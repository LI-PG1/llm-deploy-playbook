#!/bin/bash

echo "============================================"
echo "  BitCPM-CANN-8B Inference Service"
echo "  面壁智能 · OpenBMB · 1.58-bit 端侧大模型"
echo "============================================"

MODEL_DIR="/gemini/pretrain/BitCPM-CANN-8B"
export MODEL_PATH="$MODEL_DIR"

echo ""
echo "[1/3] Checking model files..."
if [ ! -d "$MODEL_DIR" ]; then
    echo "ERROR: Model directory $MODEL_DIR not found!"
    ls -la /gemini/pretrain/ 2>/dev/null
    exit 1
fi

FILE_COUNT=$(find "$MODEL_DIR" -name "*.bin" -o -name "*.safetensors" 2>/dev/null | wc -l)
echo "Found $FILE_COUNT weight files"

# 确认 pytorch_model.bin 存在
WEIGHT_FILE="$MODEL_DIR/pytorch_model.bin"
if [ -f "$WEIGHT_FILE" ]; then
    WEIGHT_SIZE=$(ls -lh "$WEIGHT_FILE" | awk '{print $5}')
    echo "  pytorch_model.bin: $WEIGHT_SIZE"
else
    echo "WARNING: pytorch_model.bin not found!"
fi

echo ""
echo "[2/3] Environment:"
echo "  torch: $(python -c 'import torch;print(torch.__version__)' 2>&1)"
echo "  transformers: $(python -c 'import transformers;print(transformers.__version__)' 2>&1)"
echo "  CUDA: $(python -c 'import torch;print(torch.version.cuda)' 2>&1)"
echo "  Model path: $MODEL_PATH"

echo ""
echo "[3/3] Starting Gradio Web UI..."
cd /gemini/code

# 前台运行（K8s 容器）
python -u app.py 2>&1 | tee /tmp/app.log

if [ $? -ne 0 ]; then
    echo "FAILED! Last 50 lines of log:"
    tail -50 /tmp/app.log
    exit 1
fi
