# ============================================
# DeepSeek-R1-Distill-Qwen-14B 通用部署 Dockerfile
# 特点：中大型模型、多卡、transformers 5.x
# 对应文档：docs/03-编写Dockerfile / docs/08-代码适配
# ============================================

FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

WORKDIR /app

# ---------- 系统依赖 ----------
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# ---------- Python 依赖（注意版本锁定） ----------
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---------- 应用代码 ----------
COPY app.py start.sh ./
RUN chmod +x start.sh

EXPOSE 7860

CMD ["bash", "start.sh"]
