RUN sed -i "s/archive.ubuntu.com/mirrors.aliyun.com/g; s/security.ubuntu.com/mirrors.aliyun.com/g" /etc/apt/sources.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        git wget unzip curl && \
    rm -rf /var/lib/apt/lists/*

RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# torch 2.4 + flash_attn 2.6.3 已在基础镜像中，只需装 transformers + 应用依赖
RUN pip install --no-cache-dir \
    transformers==4.57.1 \
    tokenizers==0.22.0 \
    huggingface-hub==0.34.0 \
    safetensors==0.4.3 \
    accelerate==0.34.2 \
    gradio==6.17.3 \
    pillow==10.2.0 \
    sentencepiece==0.2.1 \
    numpy==2.2.6 \
    einops==0.8.0 \
    tqdm \
    requests

# === 假 fast_hadamard_transform ===
RUN mkdir -p /tmp/_fht/fast_hadamard_transform && \
    printf 'import torch\ndef hadamard_transform(x, scale=1.0): return x * scale\n' > /tmp/_fht/fast_hadamard_transform/__init__.py

# === transformers 4.57 check_model_inputs 兼容补丁 ===
RUN printf 'import transformers.utils.generic as G\n_orig = G.check_model_inputs\ndef _patched(func=None):\n    if func is None:\n        return _orig\n    return _orig(func)\nG.check_model_inputs = _patched\n' > /root/miniconda3/lib/python3.10/site-packages/_startup_patch_apply.py && \
    echo "import _startup_patch_apply" > /root/miniconda3/lib/python3.10/site-packages/_startup_patch.pth

COPY . /gemini/code/

ENV MODEL_PATH=/gemini/pretrain/Keye-VL-2.0-30B-A3B
ENV HF_ENDPOINT=https://hf-mirror.com
ENV HF_HUB_DISABLE_SSL_VERIFY=1
ENV PYTHONPATH=/tmp/_fht
