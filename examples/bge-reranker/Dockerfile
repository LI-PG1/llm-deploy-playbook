RUN sed -i "s/archive.ubuntu.com/mirrors.aliyun.com/g; s/security.ubuntu.com/mirrors.aliyun.com/g" /etc/apt/sources.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        git wget unzip curl && \
    rm -rf /var/lib/apt/lists/*

RUN pip config set global.index-url https://pypi.virtaicloud.com/repository/pypi/simple

RUN pip install --no-cache-dir \
    transformers \
    huggingface-hub \
    gradio \
    pillow \
    einops \
    safetensors \
    sentencepiece \
    tiktoken \
    tqdm \
    requests

ENV MODEL_PATH=/gemini/pretrain/bge-reranker-v2-m3
ENV HF_ENDPOINT=https://hf-mirror.com
ENV HF_HUB_DISABLE_SSL_VERIFY=1
