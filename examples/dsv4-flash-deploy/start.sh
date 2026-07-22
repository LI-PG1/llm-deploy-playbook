#!/bin/bash
# ============================================================
# DeepSeek-V4-Flash vLLM Server — 启动脚本
# 使用方式:
#   bash start.sh
# ============================================================

set -e

# ===== 可配置变量（进入环境后根据实际挂载路径修改）=====
MODEL_PATH="${MODEL_PATH:-/gemini/pretrain/DeepSeek-V4-Flash-model}"
PORT=${PORT:-8000}
TP_SIZE=${TP_SIZE:-4}

# ===== 环境变量 =====
# NCCL：OrionX 环境使用 stub NCCL，需要预加载完整库
# NCCL：OrionX stub 不含 ncclDevCommCreate，预加载完整库
# 注：文件命名为 libnccl_full.s2（不含 "nccl"）以绕过 OrionX 劫持
NCCL_LIB="/usr/local/lib/libnccl_full.s2"
if [ -f "$NCCL_LIB" ]; then
    export LD_PRELOAD="$NCCL_LIB"
fi

# HuggingFace 镜像
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export HF_HUB_DISABLE_SSL_VERIFY=1

# vLLM 优化
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=256

# A100 (sm_80) 兼容
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"

echo "================================================"
echo "  DeepSeek-V4-Flash vLLM Server"
echo "  Model: $MODEL_PATH"
echo "  TP:    $TP_SIZE"
echo "  Port:  $PORT"
echo "================================================"
echo ""

# ===== Step 1: 检查模型目录 =====
echo "[1/3] Checking model directory..."
if [ ! -d "$MODEL_PATH" ]; then
    echo "[ERROR] Model directory not found: $MODEL_PATH"
    echo "[HINT] Check actual path with: ls /gemini/pretrain/"
    exit 1
fi
FILE_COUNT=$(ls "$MODEL_PATH"/*.safetensors 2>/dev/null | wc -l)
echo "[OK] Found $FILE_COUNT safetensors shards"

# ===== Step 2: 安装 DeepGEMM FP8 内核 =====
echo ""
echo "[2/3] Checking DeepGEMM FP8 kernels..."
python3 -c "import deep_gemm" 2>/dev/null && {
    echo "[OK] DeepGEMM already installed"
} || {
    echo "[...] Installing DeepGEMM..."
    bash <(curl -fsSL https://raw.githubusercontent.com/vllm-project/vllm/main/tools/install_deepgemm.sh) 2>&1 || {
        echo "[WARN] DeepGEMM install failed (may be bundled in vLLM 0.24.0)"
    }
}

# ===== Step 3: 启动 vLLM Server =====
echo ""
echo "[3/3] Starting vLLM server..."
echo ""

exec vllm serve "$MODEL_PATH" \
    --trust-remote-code \
    --tensor-parallel-size "$TP_SIZE" \
    --kv-cache-dtype fp8 \
    --block-size 256 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.92 \
    --max-num-seqs 64 \
    --max-num-batched-tokens 8192 \
    --tokenizer-mode deepseek_v4 \
    --tool-call-parser deepseek_v4 \
    --enable-auto-tool-choice \
    --reasoning-parser deepseek_v4 \
    --served-model-name deepseek-v4-flash \
    --port "$PORT" \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --distributed-executor-backend mp \
    2>&1 | tee /tmp/vllm_flash.log
