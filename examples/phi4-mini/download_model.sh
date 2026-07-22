#!/bin/bash
# 下载 Phi-4-mini 模型权重
# 用法: bash download_model.sh /path/to/save

MODEL_DIR=${1:-./models/phi4-mini}
mkdir -p $MODEL_DIR

echo "下载 Phi-4-mini-instruct..."
pip install huggingface_hub -q
huggingface-cli download microsoft/Phi-4-mini-instruct   --local-dir $MODEL_DIR   --local-dir-use-symlinks False

echo "下载完成: $MODEL_DIR"
echo "模型大小: $(du -sh $MODEL_DIR | cut -f1)"
