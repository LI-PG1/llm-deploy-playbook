#!/bin/bash
export LD_LIBRARY_PATH=$(find /opt/orion -name "libcudnn.so.9" 2>/dev/null | head -1 | xargs dirname):$LD_LIBRARY_PATH
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_SSL_VERIFY=1
export TRANSFORMERS_OFFLINE=1

# 模型目录含点号，Python import 会解析失败，建软链避开
export MODEL_PATH=/gemini/pretrain/Laguna-XS.2
if [ ! -L /tmp/LagunaXS2 ]; then
    ln -sf $MODEL_PATH /tmp/LagunaXS2
fi
export MODEL_PATH=/tmp/LagunaXS2

echo "[start] CUDA: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>&1 | head -1)"
echo "[start] MODEL_PATH=$MODEL_PATH"

cd /gemini/code
python -u app.py
