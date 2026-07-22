# ============================================
# gte-Qwen2-1.5B-Instruct Embedding 模型镜像
# 基础镜像：CUDA12.1.1-PyTorch2.1.1-conda3.10-ubuntu22.04
#
# 已验证配置（开发环境跑通 2026-06-02）：
#   torch=2.5.1  transformers=4.46.3  tokenizers>=0.20,<0.21
#   huggingface-hub>=0.23,<0.27  safetensors>=0.4.3
#   tokenization_qwen.py（LlamaTokenizer兼容桥，随代码上传）
#
# 关键教训引用：
#   #62 #64 所有 pip install 必须 --no-deps
#   #66 torch 2.5.1 需 cuDNN 9 → LD_LIBRARY_PATH 指向 orion-cuda-12.2
#   #67 公网包加 --retries 15 --timeout 120
#   #82 start.sh 用 exec（K8s 推理主进程）
#   #89 start.sh 用 tee 双写日志
# ============================================

RUN sed -i "s/archive.ubuntu.com/mirrors.aliyun.com/g; s/security.ubuntu.com/mirrors.aliyun.com/g" /etc/apt/sources.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        git wget unzip curl && \
    rm -rf /var/lib/apt/lists/*

RUN pip config set global.index-url https://pypi.virtaicloud.com/repository/pypi/simple

# ==================== 公网 PyPI 包（内网版本不够） ====================
RUN pip install --no-cache-dir --no-deps \
    -i https://pypi.org/simple/ \
    --trusted-host pypi.org --trusted-host files.pythonhosted.org \
    --retries 15 --timeout 120 \
    "transformers==4.46.3"

RUN pip install --no-cache-dir --no-deps \
    -i https://pypi.org/simple/ \
    --trusted-host pypi.org --trusted-host files.pythonhosted.org \
    --retries 15 --timeout 120 \
    "tokenizers>=0.20,<0.21"

RUN pip install --no-cache-dir --no-deps \
    -i https://pypi.org/simple/ \
    --trusted-host pypi.org --trusted-host files.pythonhosted.org \
    --retries 15 --timeout 120 \
    "huggingface-hub>=0.23,<0.27"

RUN pip install --no-cache-dir --no-deps \
    -i https://pypi.org/simple/ \
    --trusted-host pypi.org --trusted-host files.pythonhosted.org \
    --retries 15 --timeout 120 \
    "regex>=2023,<2025"

# ==================== 内网 PyPI 包 ====================
RUN pip install --no-cache-dir --force-reinstall --no-deps \
    --trusted-host pypi.virtaicloud.com \
    --retries 15 --timeout 120 \
    torch==2.5.1

RUN pip install --no-cache-dir --force-reinstall --no-deps \
    --trusted-host pypi.virtaicloud.com \
    --retries 15 --timeout 120 \
    "safetensors>=0.4.3"

RUN pip install --no-cache-dir --no-deps \
    --trusted-host pypi.virtaicloud.com \
    --retries 15 --timeout 120 \
    gradio einops numpy tqdm requests accelerate

# ==================== tokenization_qwen.py（LlamaTokenizer 兼容桥） ====================
RUN printf 'from transformers import LlamaTokenizer\n\nclass Qwen2Tokenizer(LlamaTokenizer):\n    pass\n' > /tmp/tokenization_qwen.py

ENV MODEL_PATH=/gemini/pretrain/gte-Qwen2-1.5B-instruct
ENV HF_ENDPOINT=https://hf-mirror.com
ENV HF_HUB_DISABLE_SSL_VERIFY=1
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ENV LD_LIBRARY_PATH=/opt/orion/orion_runtime/gpu/cuda/orion-cuda-12.2/:/usr/local/cuda/lib64:/root/miniconda3/lib
