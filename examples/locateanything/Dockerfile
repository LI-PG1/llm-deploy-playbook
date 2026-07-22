# LocateAnything-3B 推理服务
# 基础镜像：Pytorch 2.x · Python 3.11 · Ubuntu 22.04 · Cuda 12.x · CuDNN 9

# ─── 全依赖安装（不带 --no-deps，避免缺子依赖如 gradio_client）────
# 版本锁定依据：2026-06-15 开发环境已验证通过
RUN pip install --no-cache-dir \
    gradio==6.17.3 \
    pillow==11.0.0 \
    einops==0.8.2 \
    safetensors==0.8.0 \
    numpy==2.4.6 \
    tqdm==4.65.0 \
    requests==2.31.0 \
    opencv-python-headless==4.13.0.92 \
    peft==0.19.1 \
    accelerate==1.14.0 \
    decord==0.6.0 \
    lmdb==2.2.1 \
    transformers==4.57.1 \
    huggingface_hub==0.36.2 \
    tokenizers==0.22.2 \
    --index-url https://pypi.tuna.tsinghua.edu.cn/simple/

# ─── 拷贝代码文件（与 Dockerfile 同目录上传）──────────────────────────
RUN mkdir -p /gemini/code
COPY app.py start.sh /gemini/code/
RUN chmod +x /gemini/code/start.sh

# ─── Environment ────────────────────────────────────────────────────
ENV MODEL_PATH=/gemini/pretrain/LocateAnything-3B-model
ENV LD_LIBRARY_PATH=/opt/orion/orion_runtime/gpu/cuda/orion-cuda-12.4/
ENV HF_ENDPOINT=https://hf-mirror.com
ENV HF_HUB_DISABLE_SSL_VERIFY=1
ENV TRANSFORMERS_OFFLINE=1
ENV HF_HUB_DISABLE_SYMLINKS_WARNING=1
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860
