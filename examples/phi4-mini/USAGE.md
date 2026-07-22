# Phi-4-mini 部署示例

## 文件说明

| 文件 | 作用 |
|------|------|
| app.py | Gradio 聊天界面，线程安全加载模型 |
| Dockerfile | 最小可跑镜像（pytorch:2.5.1-cuda12.4） |
| requirements.txt | 依赖锁定 |
| start.sh | 容器启动入口 |
| download_model.sh | 模型权重下载脚本 |

## 使用步骤

```bash
# 1. 下载模型
bash download_model.sh ./models/phi4-mini

# 2. 修改 app.py 中的 MODEL_PATH 为实际路径
# 3. 构建镜像
docker build -t phi4-mini-demo .

# 4. 运行
docker run --gpus all -p 7860:7860 \
  -v ./models:/app/models \
  phi4-mini-demo
```

## 注意事项
- 模型约 7.7GB（BF16），需 12GB+ 显存
- 支持 INT4 量化加载（load_in_4bit=True）
