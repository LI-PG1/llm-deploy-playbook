#!/bin/bash
#
# DiffusionGemma 26B-A4B-it 推理服务启动脚本
# 适用场景：趋动云推理服务（K8s 前台进程）
#

echo "============================================"
echo "  DiffusionGemma 26B-A4B-it — 推理服务"
echo "============================================"

# ── 环境变量 ──────────────────────────────────────
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_SSL_VERIFY=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

echo ""
echo "============================================"
echo "  BEFORE FIRST RUN:"
echo "  Run 'ls /gemini/pretrain/' to find the"
echo "  actual model directory name, then update"
echo "  MODEL_DIR below if it differs."
echo "============================================"

# ── 模型路径 ──────────────────────────────────────
MODEL_DIR="/gemini/pretrain/diffusiongemma-26B-A4B-it"
export MODEL_PATH="$MODEL_DIR"

mkdir -p /tmp/output

echo ""
echo "[1/3] Checking model files..."
if [ -d "$MODEL_DIR" ]; then
    shard_count=$(ls "$MODEL_DIR"/model-0*.safetensors 2>/dev/null | wc -l)
    config_ok=$(ls "$MODEL_DIR"/config.json 2>/dev/null | wc -l)
    echo "[OK] Model directory: $MODEL_DIR"
    echo "[OK] safetensors shards: $shard_count"
    echo "[OK] config.json: $config_ok"
    if [ "$shard_count" -lt 11 ] || [ "$config_ok" -eq 0 ]; then
        echo "[ERROR] Model files incomplete! Expected 11 shards + config.json"
        ls -la "$MODEL_DIR"/ 2>/dev/null || echo "  (directory not accessible)"
        exit 1
    fi
else
    echo "[ERROR] Model directory not found: $MODEL_DIR"
    echo "[HINT] Run: ls /gemini/pretrain/ to check actual directory name"
    exit 1
fi

echo ""
echo "[2/3] Checking system..."
echo "[DIAG] python:  $(which python)"
echo "[DIAG] torch:   $(python -c 'import torch; print(torch.__version__)' 2>&1)"
echo "[DIAG] trans:   $(python -c 'import transformers; print(transformers.__version__)' 2>&1)"
echo "[DIAG] gradio:  $(python -c 'import gradio; print(gradio.__version__)' 2>&1)"
echo "[DIAG] CUDA:    $(python -c 'import torch; print(torch.cuda.is_available())' 2>&1)"
if python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null | grep -q True; then
    echo "[DIAG] GPU:     $(python -c "
import torch; d=torch.cuda.get_device_name(0)
v=torch.cuda.get_device_properties(0).total_memory/1e9
print(f'{d} | {v:.1f} GB VRAM')
" 2>&1)"
fi

echo ""
echo "[3/3] Starting DiffusionGemma WebUI..."
cd /gemini/code

# 推理服务要求前台进程（不可用 nohup）
export PYTHONUNBUFFERED=1
echo "  Listening on 0.0.0.0:7860"
echo "  Model: $MODEL_DIR"
echo ""

python -u app.py 2>&1 | tee /tmp/app.log
