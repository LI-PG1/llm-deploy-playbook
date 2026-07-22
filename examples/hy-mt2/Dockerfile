RUN sed -i "s/archive.ubuntu.com/mirrors.aliyun.com/g; s/security.ubuntu.com/mirrors.aliyun.com/g" /etc/apt/sources.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        git wget unzip curl && \
    rm -rf /var/lib/apt/lists/*

RUN pip config set global.index-url https://pypi.virtaicloud.com/repository/pypi/simple

RUN pip install --no-cache-dir --no-deps \
    gradio \
    pillow \
    einops \
    safetensors \
    sentencepiece \
    tiktoken \
    tqdm \
    requests

RUN pip install --no-cache-dir --no-deps \
    "torch>=2.4.0" "torchvision>=0.19.0" "torchaudio>=2.4.0" \
    --index-url https://download.pytorch.org/whl/cu121

RUN pip install --no-cache-dir --no-deps \
    "tokenizers>=0.22.0,<=0.23.0" \
    "safetensors>=0.4.3" \
    "regex>=2025.10.22" \
    "accelerate>=1.0" \
    "huggingface-hub>=1.5.0,<2.0" \
    "transformers==5.6.0" \
    --index-url https://pypi.tuna.tsinghua.edu.cn/simple/

RUN printf '\ndef is_offline_mode():\n    return False\n' >> /root/miniconda3/lib/python3.10/site-packages/huggingface_hub/__init__.py

ENV MODEL_PATH=/gemini/pretrain/Hy-MT2-7B-model
ENV LD_LIBRARY_PATH=/opt/orion/orion_runtime/gpu/cuda/orion-cuda-12.4/
ENV HF_ENDPOINT=https://hf-mirror.com
ENV HF_HUB_DISABLE_SSL_VERIFY=1
