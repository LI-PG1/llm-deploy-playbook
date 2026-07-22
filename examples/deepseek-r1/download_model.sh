#!/bin/bash
# 下载 DeepSeek-R1-Distill-Qwen-14B 模型权重
# 用法: bash download_model.sh /path/to/save

MODEL_DIR=${1:-./models/deepseek-r1-14b}
mkdir -p $MODEL_DIR

echo "下载 DeepSeek-R1-Distill-Qwen-14B..."
pip install huggingface_hub -q
huggingface-cli download deepseek-ai/DeepSeek-R1-Distill-Qwen-14B   --local-dir $MODEL_DIR   --local-dir-use-symlinks False

echo "下载完成: $MODEL_DIR"
echo "模型大小: $(du -sh $MODEL_DIR | cut -f1)"
