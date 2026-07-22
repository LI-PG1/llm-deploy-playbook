# DiffusionGemma 26B-A4B-it 扩散视觉语言推理服务
# 架构: DiffusionGemmaForBlockDiffusion · MoE 25.8B / 3.8B active
# 注意：无 FROM/EXPOSE/CMD，平台自动插入

# ---- 环境变量 ----
ENV MODEL_PATH=/gemini/pretrain/diffusiongemma-26B-A4B-it
ENV HF_ENDPOINT=https://hf-mirror.com
ENV HF_HUB_DISABLE_SSL_VERIFY=1
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# ---- PyTorch（官源，--no-deps 避免 NCCL 锁） ----
RUN pip install --no-cache-dir --no-deps \
    torch==2.6.0 torchvision==0.21.0 \
    --index-url https://download.pytorch.org/whl/cu124

# ---- 推理依赖（清华源） ----
RUN pip install --no-cache-dir \
    transformers==5.12.1 \
    gradio==6.19.0 \
    accelerate==1.14.0 \
    safetensors==0.8.0 \
    tokenizers==0.22.2 \
    pillow==11.0.0 \
    einops \
    numpy==1.26.4 \
    --index-url https://pypi.tuna.tsinghua.edu.cn/simple
