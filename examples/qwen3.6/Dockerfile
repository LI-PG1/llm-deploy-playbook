# ============================================
# Qwen3.6-35B-A3B 对话模型镜像 Dockerfile
# 基础镜像：CUDA12.1.1-PyTorch2.1.1-conda3.10-ubuntu22.04
#
# 已验证配置（开发环境跑通 ✅ 2026-05-27）：
#   torch=2.5.1  transformers=5.9.0  accelerate=1.13.0
#   huggingface-hub=1.16.4  tokenizers=0.22.2  regex=2026.5.9
#   nvidia-cudnn-cu12=9.22.0.52  gradio=4.7.1  einops=0.7.0
#
# 关键教训：
#   #62 所有 pip install 必须 --no-deps
#   #63 huggingface-hub 需从公网装（内网无 >=0.31）
#   #67 公网包加 --retries 15 --timeout 120
#   #70 accelerate>=1.0 必须装（FP8 量化需要）
#   #73 device_map="auto" + expandable_segments 防 OOM
# ============================================

RUN sed -i "s/archive.ubuntu.com/mirrors.aliyun.com/g; s/security.ubuntu.com/mirrors.aliyun.com/g" /etc/apt/sources.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        git wget unzip curl && \
    rm -rf /var/lib/apt/lists/*

RUN pip config set global.index-url https://pypi.virtaicloud.com/repository/pypi/simple

# ==================== 公网 PyPI 包（内网没有或版本不够） ====================
# transformers 5.x / huggingface-hub>=0.31 / tokenizers / regex / cudnn

RUN pip install --no-cache-dir --no-deps \
    -i https://pypi.org/simple/ \
    --trusted-host pypi.org --trusted-host files.pythonhosted.org \
    --retries 15 --timeout 120 \
    transformers==5.9.0

RUN pip install --no-cache-dir --no-deps \
    -i https://pypi.org/simple/ \
    --trusted-host pypi.org --trusted-host files.pythonhosted.org \
    --retries 15 --timeout 120 \
    "huggingface-hub>=0.31.0"

RUN pip install --no-cache-dir --no-deps \
    -i https://pypi.org/simple/ \
    --trusted-host pypi.org --trusted-host files.pythonhosted.org \
    --retries 15 --timeout 120 \
    "tokenizers>=0.22.0,<=0.23.0"

RUN pip install --no-cache-dir --no-deps \
    -i https://pypi.org/simple/ \
    --trusted-host pypi.org --trusted-host files.pythonhosted.org \
    --retries 15 --timeout 120 \
    "regex>=2025.10.22"

# ==================== 内网 PyPI 包 ====================

RUN pip install --no-cache-dir --force-reinstall --no-deps \
    --trusted-host pypi.virtaicloud.com \
    --retries 15 --timeout 120 \
    torch==2.5.1

RUN pip install --no-cache-dir --no-deps \
    -i https://pypi.org/simple/ \
    --trusted-host pypi.org --trusted-host files.pythonhosted.org \
    --retries 15 --timeout 120 \
    "accelerate>=1.0"

RUN pip install --no-cache-dir --force-reinstall --no-deps \
    -i https://pypi.org/simple/ \
    --trusted-host pypi.org --trusted-host files.pythonhosted.org \
    --retries 15 --timeout 120 \
    "safetensors>=0.4.3"

RUN pip install --no-cache-dir --no-deps \
    --trusted-host pypi.virtaicloud.com \
    --retries 15 --timeout 120 \
    gradio pillow einops sentencepiece tiktoken tqdm requests

ENV MODEL_PATH=/gemini/pretrain/Qwen3.6-35B-A3B
ENV HF_ENDPOINT=https://hf-mirror.com
ENV HF_HUB_DISABLE_SSL_VERIFY=1
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ENV LD_LIBRARY_PATH=/opt/orion/orion_runtime/gpu/cuda/orion-cuda-12.2/:/usr/local/cuda/lib64:/root/miniconda3/lib
