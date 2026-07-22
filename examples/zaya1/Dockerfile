# ============================================
# ZAYA1-8B MoE 推理服务镜像 Dockerfile
# 基础镜像：CUDA12.1.1-PyTorch2.1.1-conda3.10-ubuntu22.04
#
# transformers-zaya1：Zyphra 自定义 fork v4.57.1
#   托管：https://gitee.com/hackerca/zaya/releases
# ============================================

RUN sed -i "s/archive.ubuntu.com/mirrors.aliyun.com/g; s/security.ubuntu.com/mirrors.aliyun.com/g" /etc/apt/sources.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        git wget unzip curl && \
    rm -rf /var/lib/apt/lists/*

RUN pip config set global.index-url https://pypi.virtaicloud.com/repository/pypi/simple

# ========== 公网 PyPI 包 ==========

RUN pip install --no-cache-dir --no-deps \
    -i https://pypi.org/simple/ \
    --trusted-host pypi.org --trusted-host files.pythonhosted.org \
    --retries 15 --timeout 120 \
    "huggingface-hub>=0.34.0,<1.0"

RUN pip install --no-cache-dir --no-deps \
    -i https://pypi.org/simple/ \
    --trusted-host pypi.org --trusted-host files.pythonhosted.org \
    --retries 15 --timeout 120 \
    "tokenizers>=0.22.0,<=0.23.0"

RUN pip install --no-cache-dir --no-deps \
    -i https://pypi.org/simple/ \
    --trusted-host pypi.org --trusted-host files.pythonhosted.org \
    --retries 15 --timeout 120 \
    "safetensors>=0.4.3"

RUN pip install --no-cache-dir --no-deps \
    -i https://pypi.org/simple/ \
    --trusted-host pypi.org --trusted-host files.pythonhosted.org \
    --retries 15 --timeout 120 \
    "regex>=2025.10.22"

RUN pip install --no-cache-dir --no-deps \
    -i https://pypi.org/simple/ \
    --trusted-host pypi.org --trusted-host files.pythonhosted.org \
    --retries 15 --timeout 120 \
    "accelerate>=1.0"

# torch >= 2.2 required by transformers-zaya1 (base image has 2.1.1)
RUN pip install --no-cache-dir --force-reinstall --no-deps \
    --trusted-host pypi.virtaicloud.com \
    --retries 15 --timeout 120 \
    torch==2.2.2

# ========== 自定义 transformers (Zyphra fork v4.57.1) ==========

RUN wget -q --user-agent="Mozilla/5.0" \
    "https://gitee.com/hackerca/zaya/releases/download/v1.0/transformers-zaya1.tar.gz" \
    -O /tmp/transformers-zaya1.tar.gz && \
    pip install --no-cache-dir --no-deps /tmp/transformers-zaya1.tar.gz && \
    rm -f /tmp/transformers-zaya1.tar.gz

# ========== 内网 PyPI 包 ==========

RUN pip install --no-cache-dir --no-deps \
    --trusted-host pypi.virtaicloud.com \
    --retries 15 --timeout 120 \
    gradio pillow einops sentencepiece tiktoken tqdm requests

# ========== 环境变量 ==========

ENV MODEL_PATH=/gemini/pretrain
ENV HF_ENDPOINT=https://hf-mirror.com
ENV HF_HUB_DISABLE_SSL_VERIFY=1
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ENV LD_LIBRARY_PATH=/opt/orion/orion_runtime/gpu/cuda/orion-cuda-12.2/:/usr/local/cuda/lib64:/root/miniconda3/lib
