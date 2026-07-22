# ============================================
# Supertone-3 TTS 模型镜像 Dockerfile
# 基础镜像：PyTorch2.1.1-CUDA12.1.1-Python3.10-Ubuntu22.04
#
# 策略：
#   - apt: aliyun 源
#   - pip: 平台内网 PyPI
#   - supertonic: pip 安装（ONNX CPU推理，无需GPU）
#   - 模型: 构建时从 HF-Mirror 下载并缓存进镜像
# ============================================

RUN sed -i "s/archive.ubuntu.com/mirrors.aliyun.com/g; s/security.ubuntu.com/mirrors.aliyun.com/g" /etc/apt/sources.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        git wget \
        libsndfile1 && \
    rm -rf /var/lib/apt/lists/*

RUN pip config set global.index-url https://pypi.virtaicloud.com/repository/pypi/simple

RUN pip install --no-cache-dir supertonic gradio==4.44.1 numpy==1.25.2 && \
    pip install --no-cache-dir "huggingface-hub<1.0" && \
    HF_ENDPOINT=https://hf-mirror.com python -c "from huggingface_hub import snapshot_download; snapshot_download('Supertone/supertonic-3')"

ENV GRADIO_SERVER_NAME=0.0.0.0
ENV HF_ENDPOINT=https://hf-mirror.com
