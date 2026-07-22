#!/bin/bash
echo "============================================"
echo "  Kronos 金融K线预测服务"
echo "============================================"

export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_SSL_VERIFY=1

MODEL_DIR="/gemini/pretrain"
export MODEL_PATH="$MODEL_DIR"

mkdir -p /tmp/output

echo ""
echo "[1/3] Checking model files..."
for dir in "Kronos-Tokenizer-base" "Kronos-small" "Kronos-base"; do
    if [ -d "$MODEL_DIR/$dir" ]; then
        files=$(ls "$MODEL_DIR/$dir"/*.safetensors 2>/dev/null | wc -l)
        echo "  [OK] $dir ($files safetensors)"
    else
        echo "  [WARN] $dir not found"
    fi
done

echo ""
echo "[2/3] Checking/Installing dependencies..."
pip install -q gradio pandas numpy matplotlib huggingface-hub torch 2>/dev/null

echo ""
echo "[3/3] Starting Gradio WebUI..."
cd /gemini/code
exec python app.py 2>&1 | tee /tmp/app.log
