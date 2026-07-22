# llm-deploy-playbook

大模型部署实战笔记。包含多个模型从 Docker 容器化到推理服务化的完整示例。

## 快速开始

```bash
# 选一个示例，比如 phi4-mini
cd examples/phi4-mini

# 看使用说明
cat USAGE.md

# 下载模型权重
bash download_model.sh

# 构建镜像并运行
docker build -t phi4-mini-demo .
docker run --gpus all -p 7860:7860 phi4-mini-demo
```

## 项目一览

examples/ 目录下有 27 个已部署模型的完整工程文件，按场景分类：

| 场景 | 模型 |
|------|------|
| 文本生成 / 对话 | deepseek-r1-14b, phi4-mini, qwen3.5-122b, qwen3.6, qwen3-coder, contentv-8b |
| MoE 大模型 | qwen3.5-122b, qwen3.6, agentworld-35b-a3b, dsv4-flash |
| Embedding / Rerank | gte-qwen2, bge-reranker |
| 多模态 / VLM | keye-vl2, gemma4-12b, locateanything, minicpm5, starvla |
| 语音 / TTS | supertone3, stable-audio3, hy-mt2, sulphur2 |
| 图像 / 视频 | diffusiongemma, zimage-turbo, laguna-xs2 |
| OCR | unlimited-ocr, kronos |
| 其他 | zaya1, bitcpm-cann-8b |

每个示例目录包含：

```
- app.py              主程序（Gradio / FastAPI）
- Dockerfile          容器镜像
- requirements.txt    依赖锁定
- start.sh            启动脚本
- download_model.sh   模型权重下载
- USAGE.md            使用说明
```

## 相关仓库

- [llm-model-optimization](https://github.com/LI-PG1/llm-model-optimization) — 量化基准测试和微调模板
- [llm-rag-agent-service](https://github.com/LI-PG1/llm-rag-agent-service) — RAG + Agent 融合服务
