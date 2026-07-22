# llm-deploy-playbook

大模型部署实战笔记。整理了在云平台交付多个模型的过程中遇到的问题和解决方案。

---

## 目录

* [README.md](./README.md)
* [examples/](./examples/) — 各类型模型的完整部署示例（共 27 个项目）

| 类别 | 包含 |
|------|------|
| 文本生成 / MoE | deepseek-r1, qwen3.5-122b, qwen3.6, phi4-mini, agentworld, contentv, starvla |
| Embedding & Reranker | gte-qwen2, bge-reranker |
| 多模态 / VLM | keye-vl2, gemma4-12b, locateanything, minicpm5 |
| 语音 / 音频 | supertone3, stable-audio3, hy-mt2 |
| 图像 / 视频生成 | diffusiongemma, zimage-turbo, laguna-xs2 |
| OCR 文档 | unlimited-ocr, kronos |
| 其他 | dsv4-flash, sulphur2, zaya1, bitcpm-8b, qwen3-coder |

## 相关仓库

* [llm-model-optimization](https://github.com/LI-PG1/llm-model-optimization) — 量化基准测试和微调模板
* [llm-rag-agent-service](https://github.com/LI-PG1/llm-rag-agent-service) — RAG + Agent 融合服务
