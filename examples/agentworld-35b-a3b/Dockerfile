# ============================================
# Qwen-AgentWorld-35B-A3B 推理服务镜像
# Agent 环境模拟器 · 7 领域
# ============================================
# 基础镜像：公共基础镜像
# Python 3.11 · CUDA 12.1 · PyTorch 2.2.2
# 已验证环境：torch 2.5.1 · transformers 5.9.0 · gradio 6.17.3

# 注：基础镜像已预装 curl/wget/git，无需 APT 安装
#     平台内网 APT 源 mirrors.virtaicloud.com 不稳定（500）

# ---- pip 源 ----
RUN pip config set global.index-url https://pypi.virtaicloud.com/repository/pypi/simple

# ---- 升级 PyTorch 到 2.5.1（兼容 transformers 5.9.0） ----
# 不设 --no-deps：让 nvidia-* 子包（nvidia-curand-cu12 等）一并安装，
# 避免后续 accelerate 依赖解析时出现版本冲突
RUN pip install torch==2.5.1 torchvision==0.20.1 \
    --index-url https://download.pytorch.org/whl/cu121 \
    --retries 15 --timeout 120

# ---- 项目核心依赖（不加 --no-deps：Docker 构建无 NCCL 锁问题，#114） ----
RUN pip install \
    transformers==5.9.0 \
    gradio==6.17.3 \
    accelerate==1.13.0 \
    huggingface_hub==1.18.0 \
    safetensors==0.8.0 \
    tokenizers==0.22.2 \
    einops==0.8.2 \
    numpy==1.26.4

# ---- 环境变量 ----
ENV HF_ENDPOINT=https://hf-mirror.com
ENV HF_HUB_DISABLE_SSL_VERIFY=1
ENV TRANSFORMERS_OFFLINE=1
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# NCCL 路径由 start.sh 动态处理（优先使用 torch 自带 NCCL）
