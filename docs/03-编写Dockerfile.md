# 03 · 编写 Dockerfile

依赖版本确认后，把它们锁进一个可复现的镜像：基础镜像、Python 依赖、CUDA/cuDNN 版本全部写死，避免"在我机器上能跑"。模型权重不打进镜像，运行时通过挂载卷或下载到挂载目录提供。

---

## 一、决策：选哪个基础镜像

| 场景 | 推荐镜像 | 体积 | 说明 |
| :--- | :--- | :---: | :--- |
| CUDA 推理 / 训练 | `nvidia/cuda:12.x.x-cudnn-runtime-ubuntu22.04` | ~1.5 GB | NVIDIA 官方，CUDA+cuDNN 齐全 |
| 纯 PyTorch 环境 | `pytorch/pytorch:2.x.x-cuda12.x-cudnn8-runtime` | ~4 GB | 预装 PyTorch，减少构建时间 |
| CPU 推理 | `python:3.11-slim` | ~120 MB | 最小 Python 环境 |
| 开发调试 | `nvidia/cuda:12.x.x-cudnn-devel-ubuntu22.04` | ~3 GB | 含编译工具链 |

**选型原则**：推理优先 `runtime` 版；不要从 `python:3.11` 往 CUDA 容器改造（手动装 CUDA 库易版本错配且不可复现）。

## 二、标准做法：最小可跑 Dockerfile

```dockerfile
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

WORKDIR /app

# 系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl wget && rm -rf /var/lib/apt/lists/*

# Python 依赖（版本锁定！）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 应用代码
COPY app.py start.sh ./
RUN chmod +x start.sh

EXPOSE 7860
CMD ["bash", "start.sh"]
```

> **依赖安装策略**：先在开发环境逐个依赖调通、确认版本，再抄进 Dockerfile，可节约大量试错时间。内网源的版本可能与 PyPI 官方不同，新包第一次构建先**不指定版本**看实际装了什么，再锁定回去。

---

## 三、常见坑（🔧）

| 现象 | 原因 | 方案 |
| :--- | :--- | :--- |
| 构建极慢 | 构建上下文含大文件（权重/数据集/.git） | 用 `.dockerignore` 排除 `*.safetensors` `models/` `data/` 等 |
| 缓存全部失效 | `COPY . .` 放在前面 | 先把 `requirements.txt` 单独 `COPY` |
| 有 GPU 进程时 pip 报 OS Error | NCCL 锁 | 装包时加 `--no-deps` |
| 密钥泄露 | 硬编码 `HF_TOKEN` | 运行时 `docker run -e HF_TOKEN=...` 传入 |
| 容器启动慢被判定不健康 | 模型加载数分钟，健康检查过早 | HEALTHCHECK 设 `--start-period=120s` |

---

## 四、进阶速查（大表沉底）

### 构建缓存策略（按变更频率排序）

```dockerfile
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04   # 1. 几乎不变
RUN apt-get update && apt-get install -y git wget curl && rm -rf /var/lib/apt/lists/*  # 2. 很少变
COPY requirements.txt .                         # 3. 依赖变更时变
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py .                                   # 4. 最常变 → 放最后
CMD ["python", "app.py"]
```

### 安全最佳实践

```dockerfile
# 不以 root 运行
RUN addgroup --system app && adduser --system --ingroup app app
USER app
WORKDIR /home/app

# 不使用硬编码密钥
# ❌ ENV HF_TOKEN=hf_xxxxxxxxxxxx
# ✅ docker run -e HF_TOKEN=<你的 token> my-image

# 健康检查（给模型加载留足时间）
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s \
    CMD curl -f http://localhost:8000/health || exit 1
```

### pip 国内镜像源

```dockerfile
RUN pip install --no-cache-dir torch==2.5.1 \
    -i https://pypi.tuna.tsinghua.edu.cn/simple/
```

常用源：清华 `pypi.tuna.tsinghua.edu.cn`、阿里云 `mirrors.aliyun.com`、中科大 `pypi.mirrors.ustc.edu.cn`。官方源超时时设 `pip install xxx --default-timeout=120`。

### 镜像瘦身

- `.dockerignore` 排除权重文件、数据集、缓存
- 合并 `RUN` 指令降层数；删除 apt/pip 缓存（`rm -rf /var/lib/apt/lists/*`、`--no-cache-dir`）
- **多阶段构建**：编译环境单独阶段，最终镜像只拷产物

```dockerfile
FROM nvidia/cuda:12.2.0-devel-ubuntu22.04 AS builder
RUN apt-get update && apt-get install -y make gcc g++ && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir flash-attn
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04
COPY --from=builder /usr/local/lib/python3.11/dist-packages /usr/local/lib/python3.11/dist-packages
COPY app.py /app/
WORKDIR /app
CMD ["python", "app.py"]
```

### 常用指令

| 指令 | 简介 | 示例 |
| :--- | :--- | :--- |
| `COPY` | 复制本地文件到镜像 | `COPY ./app /opt/app` |
| `ADD` | 复制并支持 URL / 自动解压 | `ADD html.tar /var/www` |
| `EXPOSE` | 暴露容器端口 | `EXPOSE 80` |
| `VOLUME` | 创建挂载点（相当于 `-v`） | `VOLUME ["/data"]` |
| `WORKDIR` | 指定执行目录（不存在则创建） | `WORKDIR /opt` |
| `CMD` | 启动命令，可被 `docker run` 参数覆盖 | `CMD ["python", "app.py"]` |
| `ENTRYPOINT` | 入口程序，**不可**被覆盖 | `ENTRYPOINT ["python"]` |

### 实用技巧

- **修改已构建镜像**：基于已构建镜像叠加新层修复，不用重建整个镜像。
- **非标准包编译卡死**：用假包 stub 替代（创建同名 `__init__.py` 返回 stub 函数），配合 `PYTHONPATH` 注入。
- **不要设全局 pip 源**：`pip config set global.index-url` 会导致所有包从同一源下，遇到缺包全失败。每个 `RUN pip install` 显式指定 `--index-url`。
- **文件完整性**：大文件下载后建议检查文件大小是否接近预期，损坏的文件加载时会报奇怪的错误。
