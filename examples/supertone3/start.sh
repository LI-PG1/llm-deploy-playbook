#!/bin/bash

echo "============================================"
echo "  Supertone-3 TTS Service"
echo "============================================"

export HF_ENDPOINT=https://hf-mirror.com

pip install -q numpy==1.25.2 2>/dev/null
mkdir -p /tmp/output

echo ""
echo "[1/1] Starting Gradio Web UI..."
cd /gemini/code
python app.py --server_name 0.0.0.0 --server_port 7860 &
GRADIO_PID=$!

echo ""
echo "============================================"
echo "  Service started!"
echo "  Gradio PID: $GRADIO_PID (port 7860)"
echo "============================================"

wait
