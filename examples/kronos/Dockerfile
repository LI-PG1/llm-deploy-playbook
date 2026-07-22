# ============================================
# Kronos 金融K线预测服务 镜像
# 基础镜像：CUDA12.1.1-PyTorch2.1.1-conda3.10-ubuntu22.04
#
# 已验证配置（开发环境跑通 2026-06-11）：
#   torch 2.1.1（镜像自带） gradio 4.7.1
#   pandas numpy matplotlib huggingface-hub
#
# 关键教训引用：
#   #23 #48 HF_ENDPOINT / HF_HUB_DISABLE_SSL_VERIFY 环境变量全局设置
#   #74 from_pretrained 必须 local_files_only=True
#   #89 start.sh 用 tee 双写日志
#   #103 monkey-patch 不生效，改用 safetensors 直接加载本地模型
# ============================================

# === 系统源 & 基础工具 ===
RUN sed -i "s/archive.ubuntu.com/mirrors.aliyun.com/g; s/security.ubuntu.com/mirrors.aliyun.com/g" /etc/apt/sources.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        git wget unzip curl \
        fonts-noto-cjk && \
    rm -rf /var/lib/apt/lists/*

# === Python 镜像源 ===
RUN pip config set global.index-url https://pypi.virtaicloud.com/repository/pypi/simple

# === Python 依赖 ===
RUN pip install --no-cache-dir \
    pandas>=1.5.0 \
    numpy>=1.24.0 \
    matplotlib>=3.7.0 \
    huggingface-hub>=0.20.0 \
    safetensors>=0.4.0 \
    --retries 15 --timeout 120

RUN pip install --no-cache-dir \
    gradio>=4.0.0 \
    --retries 15 --timeout 120

# === 克隆 Kronos 模型代码（Gitee 镜像，GitHub 被墙） ===
RUN git clone --depth 1 https://gitee.com/persiacat/Kronos.git /tmp/kronos_code

# === 环境变量 ===
ENV HF_ENDPOINT=https://hf-mirror.com
ENV HF_HUB_DISABLE_SSL_VERIFY=1
ENV MODEL_PATH=/gemini/pretrain
ENV PYTHONUNBUFFERED=1
ENV GRADIO_SERVER_PORT=7860
ENV GRADIO_ANALYTICS_ENABLED=False
