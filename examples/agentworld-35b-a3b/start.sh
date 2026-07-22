#!/bin/bash
# ============================================
# Qwen-AgentWorld-35B-A3B - World Model Service
# ============================================

echo "============================================"
echo "  Qwen-AgentWorld-35B-A3B Simulator"
echo "============================================"

# NCCL 路径：优先使用 torch 自带的 NCCL（含 ncclCommResume 符号），
# 再 fallback 到 orion 运行时 CUDA 库
TORCH_NCCL_PATH="/root/miniconda3/lib/python3.11/site-packages/nvidia/nccl/lib/"
ORION_CUDA_PATH="/opt/orion/orion_runtime/gpu/cuda/orion-cuda-12.2/"
export LD_LIBRARY_PATH="${TORCH_NCCL_PATH}:${ORION_CUDA_PATH}:$LD_LIBRARY_PATH"
export TRANSFORMERS_OFFLINE=1
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_SSL_VERIFY=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ---- NCCL 修复：镜像 --no-deps 构建导致 nvidia-nccl-cu12 缺失 ----
_NCCL_LIB="/root/miniconda3/lib/python3.11/site-packages/nvidia/nccl/lib/libnccl.so.2"
if [ ! -f "$_NCCL_LIB" ]; then
    echo "[PREFLIGHT] nvidia-nccl-cu12 未安装，正在从内网 PyPI 安装..."
    pip install nvidia-nccl-cu12 --no-deps 2>&1
    if [ -f "$_NCCL_LIB" ]; then
        echo "[PREFLIGHT] ✅ nvidia-nccl-cu12 安装成功"
    else
        echo "[PREFLIGHT] ⚠️  安装失败，尝试从 PyTorch 官方源安装..."
        pip install nvidia-nccl-cu12 --no-deps \
            --index-url https://download.pytorch.org/whl/cu121 2>&1
    fi
else
    echo "[PREFLIGHT] ✅ nvidia-nccl-cu12 已就绪"
fi
# ---- NCCL 修复结束 ----

# Fake distributed env to bypass orion RPC check
export RANK=0
export WORLD_SIZE=1
export LOCAL_RANK=0
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500

MODEL_DIR="/gemini/pretrain/models"
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
echo "[2/2] Starting Gradio WebUI (port 7860)..."
cd /gemini/code

exec python -c "
import os, subprocess, sys
os.environ['RANK']='0'; os.environ['WORLD_SIZE']='1'
os.environ['LOCAL_RANK']='0'; os.environ['MASTER_ADDR']='127.0.0.1'
os.environ['MASTER_PORT']='29500'
os.environ['TRANSFORMERS_OFFLINE']='1'
os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'
subprocess.run([sys.executable, 'app.py'], env=os.environ)
" 2>&1 | tee /tmp/app.log
