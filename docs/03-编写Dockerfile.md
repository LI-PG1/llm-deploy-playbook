# 编写 Dockerfile

基础镜像选型、多阶段构建、缓存优化。

## 模板
```dockerfile
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["bash", "start.sh"]
```