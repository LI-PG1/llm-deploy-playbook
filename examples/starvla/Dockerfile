# StarVLA QwenOFT - VLA机器人推理服务
# 架构: Qwen3-VL-4B-Instruct + MLP Action Head
# 注意：无 FROM/EXPOSE/CMD，平台自动插入
# 基础镜像: CUDA12.4-PyTorch-2.6.0 (在平台 UI 中选择)
# Python 3.11, Ubuntu 22.04, CuDNN 9
# 基础镜像已预装: torch 2.6.0+cu124, torchvision 0.21.0, numpy 1.26.4, pillow 11.0.0

# ---- 环境变量 ----
ENV VLM_PATH=/gemini/pretrain/models/Qwen3-VL-4B-Instruct
ENV CKPT_PATH=/gemini/pretrain/models/StarVLA/Qwen3-VL-OFT-LIBERO-4in1/checkpoints/steps_50000_pytorch_model.pt
ENV HF_ENDPOINT=https://hf-mirror.com
ENV HF_HUB_DISABLE_SSL_VERIFY=1
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ---- 推理依赖（清华源） ----
RUN pip install --no-cache-dir \
    transformers==5.12.1 \
    gradio==6.17.3 \
    accelerate==1.14.0 \
    safetensors==0.8.0 \
    tokenizers==0.22.2 \
    einops==0.8.2 \
    omegaconf==2.3.1 \
    -i https://pypi.tuna.tsinghua.edu.cn/simple \
    --trusted-host pypi.tuna.tsinghua.edu.cn

# ---- 趋势图依赖 ----
RUN pip install --no-cache-dir matplotlib -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn

# ---- 创建日志目录 ----
RUN mkdir -p /tmp/logs
