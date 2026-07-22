# DeepSeek-R1-Distill-Qwen-14B 部署示例

## 文件说明

| 文件 | 作用 |
|------|------|
| app.py | Gradio 聊天界面，含 CoT 思考过程分离 |
| Dockerfile | 多卡 BF16 镜像 |
| requirements.txt | 依赖锁定 |
| start.sh | 容器启动入口 |
| download_model.sh | 模型权重下载脚本 |

## 使用步骤

```bash
# 1. 下载模型
bash download_model.sh ./models/dsr1-14b

# 2. 修改 app.py 中 MODEL_PATH
# 3. 构建并运行
docker build -t ds-r1-demo .
docker run --gpus all -p 7860:7860 \
  -v ./models:/app/models \
  ds-r1-demo
```

## 注意事项
- BF16 需 ~28GB 显存，建议 2 张卡（TP=2）
- app.py 中已实现 CoT 思考过程正则分离
