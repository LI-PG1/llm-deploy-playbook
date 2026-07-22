# BitCPM-CANN-8B Dockerfile
# 基础镜像：Pytorch 2.x + Cuda 12.x + CuDNN 8 + Ubuntu 22.04 + Python 3.10
# torch 2.1.1（基础镜像自带，不升级 #46）

# ---- apt 源：aliyun（#4）----
RUN sed -i "s/archive.ubuntu.com/mirrors.aliyun.com/g; s/security.ubuntu.com/mirrors.aliyun.com/g" /etc/apt/sources.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        git wget unzip curl && \
    rm -rf /var/lib/apt/lists/*

# ---- pip 源：内网优先，缺的走 tuna（新版 transformers 内网可能没有 #61）----
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# ---- 安装依赖（全 --no-deps 避免 NCCL 锁 #62）----
# torch 2.1.1、einops 0.7.0、sentencepiece 0.1.99 已在基础镜像中
RUN pip install --no-cache-dir --no-deps \
    transformers==4.46.3 \
    tokenizers==0.20.3 \
    safetensors==0.4.3 \
    accelerate==0.34.2 \
    huggingface-hub==0.24.0 \
    gradio==4.7.1 \
    pillow==10.1.0 \
    numpy==1.26.2 \
    tqdm==4.65.0 \
    requests==2.31.0

# ---- 复制代码 ----
COPY . /gemini/code/

# ---- 环境变量 ----
ENV HF_ENDPOINT=https://hf-mirror.com
ENV HF_HUB_DISABLE_SSL_VERIFY=1
ENV MODEL_PATH=/gemini/pretrain/BitCPM-CANN-8B
