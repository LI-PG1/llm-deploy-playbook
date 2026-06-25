#!/bin/bash
# ============================================
# DeepSeek-R1-Distill-Qwen-14B 启动脚本
# 场景：多 GPU 中大型模型推理，~27GB (BF16)
# 使用前：确认 MODEL_PATH 指向模型目录
# ============================================

set -e

echo "============================================"
echo "  DeepSeek-R1-Distill-Qwen-14B 部署示例"
echo "============================================"

# ---------- 模型目录检查 ----------
MODEL_DIR="${MODEL_PATH:-/models/DeepSeek-R1-Distill-Qwen-14B}"
export MODEL_PATH="$MODEL_DIR"

echo "模型目录: $MODEL_DIR"

if [ -d "$MODEL_DIR" ]; then
    file_count=$(ls "$MODEL_DIR"/*.safetensors 2>/dev/null | wc -l)
    echo "[OK] 模型目录存在，safetensors 文件: $file_count"
else
    echo "[ERROR] 模型目录不存在: $MODEL_DIR"
    echo "[HINT] ls /path/to/models/ 查看实际目录名"
    exit 1
fi

# ---------- 环境诊断 ----------
echo ""
echo "--- 环境诊断 ---"
echo "Python: $(python --version 2>&1)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)' 2>&1)"
echo "CUDA可用: $(python -c 'import torch; print(torch.cuda.is_available())' 2>&1)"
echo "GPU数量: $(python -c 'import torch; print(torch.cuda.device_count())' 2>&1)"

# ---------- 启动 ----------
echo ""
echo "--- 启动 Gradio 服务 ---"
export PYTHONUNBUFFERED=1
exec python app.py 2>&1 | tee /tmp/app.log
