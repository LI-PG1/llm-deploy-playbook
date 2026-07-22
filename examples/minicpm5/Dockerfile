# ============================================
# MiniCPM5-1B 对话模型镜像 Dockerfile
# 基础镜像：CUDA12.1.1-PyTorch2.1.1-conda3.10-ubuntu22.04
#
# MiniCPM5-1B 是标准 LlamaForCausalLM 架构
# 只需 transformers >= 5.6，无需自定义 fork
# ============================================

RUN sed -i "s/archive.ubuntu.com/mirrors.aliyun.com/g; s/security.ubuntu.com/mirrors.aliyun.com/g" /etc/apt/sources.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        git wget unzip curl && \
    rm -rf /var/lib/apt/lists/*

RUN pip config set global.index-url https://pypi.virtaicloud.com/repository/pypi/simple

RUN pip install --no-cache-dir --force-reinstall --no-deps \
    torch==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu121

RUN pip install --no-cache-dir \
    -i https://pypi.org/simple/ \
    --trusted-host pypi.org \
    --trusted-host files.pythonhosted.org \
    "transformers==5.6.0"

RUN pip install --no-cache-dir \
    huggingface-hub \
    gradio \
    pillow \
    einops \
    safetensors \
    accelerate \
    tqdm \
    requests

ENV MODEL_PATH=/gemini/pretrain/MiniCPM5-1B-model
ENV HF_ENDPOINT=https://hf-mirror.com
ENV HF_HUB_DISABLE_SSL_VERIFY=1
ENV LD_LIBRARY_PATH=/opt/orion/orion_runtime/gpu/cuda/orion-cuda-12.2/:$LD_LIBRARY_PATH
