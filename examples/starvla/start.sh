#!/bin/bash
# StarVLA QwenOFT - 启动脚本
# 在趋动云平台部署时使用

set -e

# ===== 模型路径 =====
export VLM_PATH="${VLM_PATH:-/gemini/pretrain/models/Qwen3-VL-4B-Instruct}"
export CKPT_PATH="${CKPT_PATH:-/gemini/pretrain/models/StarVLA/Qwen3-VL-OFT-LIBERO-4in1/checkpoints/steps_50000_pytorch_model.pt}"

# ===== 代码路径 =====
CODE_DIR="/gemini/code"
export PYTHONPATH="${CODE_DIR}/source:${PYTHONPATH:-}"
export STARVLA_SOURCE="${CODE_DIR}/source/starVLA"

echo "=== StarVLA QwenOFT 启动 ==="
echo "VLM:       ${VLM_PATH}"
echo "Checkpoint: ${CKPT_PATH}"
echo "PYTHONPATH: ${PYTHONPATH}"
echo "GPU:       $(python3 -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")' 2>/dev/null || echo 'N/A')"

# ===== 检查模型文件 =====
echo ""
echo "=== 模型文件检查 ==="
if [ ! -f "${CKPT_PATH}" ]; then
    echo "ERROR: 未找到 checkpoint: ${CKPT_PATH}"
    exit 1
fi
CKPT_SIZE=$(stat --print=%s "${CKPT_PATH}" 2>/dev/null || python3 -c "import os; print(os.path.getsize('${CKPT_PATH}'))")
echo "Checkpoint: $(numfmt --to=iec-i ${CKPT_SIZE} 2>/dev/null || echo ${CKPT_SIZE} bytes)"

if [ ! -d "${VLM_PATH}" ]; then
    echo "ERROR: 未找到 VLM 目录: ${VLM_PATH}"
    exit 1
fi
VLM_FILES=$(ls "${VLM_PATH}"/*.safetensors 2>/dev/null | wc -l)
echo "VLM files: ${VLM_FILES} safetensors"

# ===== 检查 StarVLA 源码 =====
echo ""
echo "=== 代码检查 ==="
if [ -d "${STARVLA_SOURCE}" ]; then
    echo "starVLA package: OK (${STARVLA_SOURCE})"
else
    echo "ERROR: starVLA package not found at ${STARVLA_SOURCE}"
    echo "       Make sure source/starVLA is uploaded to /gemini/code/source/"
    exit 1
fi

# ===== 检查 GPU =====
echo ""
echo "=== GPU 状态 ==="
python3 -c "
import torch
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        name = torch.cuda.get_device_name(i)
        total = torch.cuda.get_device_properties(i).total_memory / 1024**3
        free = torch.cuda.mem_get_info(i)[0] / 1024**3
        print(f'GPU {i}: {name}  ({total:.1f} GB total, {free:.1f} GB free)')
else:
    print('WARNING: No GPU available')
"

# ===== 后台预加载模型 =====
echo ""
echo "=== 正在启动推理服务 ==="
echo "日志: /tmp/logs/app.log"
cd "${CODE_DIR}"
exec python3 -u app.py 2>&1 | tee /tmp/logs/app.log
