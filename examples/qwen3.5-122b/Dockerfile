# ============================================
# Qwen3.5-122B-A10B 推理镜像（一步构建）
# 基础镜像：CUDA12.4-PyTorch-2.6.0（自带 PyTorch 2.6.0 + CUDA 12.x）
# ============================================

RUN pip config set global.index-url https://pypi.virtaicloud.com/repository/pypi/simple

# 安装 transformers 等推理库 + gradio WebUI
RUN PYTHONDONTWRITEBYTECODE=1 pip install --no-cache-dir \
    -i https://pypi.tuna.tsinghua.edu.cn/simple \
    --trusted-host pypi.tuna.tsinghua.edu.cn \
    --retries 15 --timeout 120 \
    "transformers>=5.9.0" \
    "huggingface-hub>=1.18.0" \
    "tokenizers>=0.22.2" \
    "accelerate>=1.13.0" \
    "regex>=2025.10.22" \
    "safetensors>=0.8.0" \
    gradio pillow einops sentencepiece tiktoken tqdm requests sympy \
  && find /root/miniconda3/lib/python3.11/site-packages \
    -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true \
  && find /root/miniconda3/lib/python3.11/site-packages \
    -type f -name "*.pyc" -delete 2>/dev/null || true \
  && true

# 安装 vLLM + vLLM 运行时依赖
# 不使用 --no-deps（跳过 deps 会导致 cachetools/cloudpickle 等缺失）
# 装完后卸载 nvidia-* pip 包（系统 CUDA 已提供，无需重复占用空间）
RUN PYTHONDONTWRITEBYTECODE=1 pip install --no-cache-dir \
    -i https://pypi.tuna.tsinghua.edu.cn/simple \
    --trusted-host pypi.tuna.tsinghua.edu.cn \
    --timeout 300 \
    "vllm==0.8.5" \
    "ray>=2.9" aiohttp msgspec openai \
  && pip uninstall -y $(pip list 2>/dev/null | grep -iE '^nvidia-' | awk '{print $1}') 2>/dev/null || true \
  && find /root/miniconda3/lib/python3.11/site-packages \
    -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true \
  && find /root/miniconda3/lib/python3.11/site-packages \
    -type f -name "*.pyc" -delete 2>/dev/null || true \
  && true

ENV MODEL_PATH=/gemini/pretrain/Qwen3.5-122B-A10B
ENV HF_ENDPOINT=https://hf-mirror.com
ENV HF_HUB_DISABLE_SSL_VERIFY=1
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
