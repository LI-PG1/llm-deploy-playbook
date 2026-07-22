#!/bin/bash
#
# ContentV-8B 视频生成推理服务启动脚本
# 适用场景：趋动云推理服务（K8s 前台进程）
#

echo "============================================"
echo "  ContentV-8B — 视频生成推理服务"
echo "  ByteDance/ContentV-8B (8B T2V)"
echo "============================================"

# ── 环境变量 ──────────────────────────────────────
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_SSL_VERIFY=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

echo ""
echo "  Model path: /gemini/pretrain/ContentV-8B"
echo "============================================"

# ── 模型路径 ──────────────────────────────────────
MODEL_DIR="/gemini/pretrain/ContentV-8B"
export MODEL_PATH="$MODEL_DIR"

CODE_DIR="/gemini/code"
mkdir -p /tmp/output

echo ""
echo "[1/4] Checking model files..."
if [ -d "$MODEL_DIR" ]; then
    shard_count=$(ls "$MODEL_DIR"/transformer/*.safetensors 2>/dev/null | wc -l)
    vae_ok=$(ls "$MODEL_DIR"/vae/*.safetensors 2>/dev/null | wc -l)
    te1_ok=$(ls "$MODEL_DIR"/text_encoder/*.safetensors 2>/dev/null | wc -l)
    te3_ok=$(ls "$MODEL_DIR"/text_encoder_3/*.safetensors 2>/dev/null | wc -l)
    config_ok=$(ls "$MODEL_DIR"/model_index.json 2>/dev/null | wc -l)
    echo "[OK] Model directory: $MODEL_DIR"
    echo "[OK] Transformer shards:   $shard_count  (expect 2)"
    echo "[OK] VAE safetensors:      $vae_ok       (expect 1)"
    echo "[OK] text_encoder:         $te1_ok       (expect 1)"
    echo "[OK] text_encoder_3:       $te3_ok       (expect 2)"
    echo "[OK] model_index.json:     $config_ok"
    if [ "$shard_count" -lt 1 ] || [ "$config_ok" -eq 0 ]; then
        echo "[ERROR] Model files incomplete!"
        ls "$MODEL_DIR"/ 2>/dev/null | head -20
        exit 1
    fi
else
    echo "[ERROR] Model directory not found: $MODEL_DIR"
    echo "[HINT] Verify model mount path in platform console"
    exit 1
fi

echo ""
echo "[2/4] Checking system..."
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
echo "[3/4] Starting ContentV-8B WebUI..."
cd "$CODE_DIR"

# 推理服务要求前台进程
export PYTHONUNBUFFERED=1
echo "  Listening on 0.0.0.0:7860"
echo "  Model: $MODEL_DIR"
echo ""

echo "============================================"
echo "  Service started — please wait ~2-5 min"
echo "  for model loading to complete..."
echo "============================================"

python -u app.py 2>&1 | tee /tmp/app.log
