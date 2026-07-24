# llm-deploy-playbook

大模型部署实战笔记。包含多个模型从容器化到推理服务化的完整示例。

## examples/

27 个已部署模型的完整工程文件：

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

每个示例至少包含 app.py、Dockerfile、requirements.txt、start.sh、download_model.sh、USAGE.md 等基础工程文件，并可能按模型特性额外附带 benchmark 脚本、配置文件或说明文档


## 社区项目

**个人主页**：[趋动云 · LinusLI](https://open.virtaicloud.com/web/profile/154373/publish)

**项目展示**：

- [nvidia/LocateAnything-3B](https://mp.weixin.qq.com/s/7jOzvru2F1FDCoU6n14aVg) — 视觉定位推理服务。NVIDIA Parallel Box Decoding，混合推理模式
- [baidu/Unlimited-OCR](https://open.virtaicloud.com/web/project/detail/727383845343547392) — 端到端 OCR 推理服务。多页 PDF 一次性解析，R-SWA 恒定 KV 缓存
- [Google/Gemma-4-Unified-12B-it](https://open.virtaicloud.com/web/project/detail/732855980610412544) — 通用多模态推理服务。Gemma 4 统一架构，256K 上下文

## 相关仓库

- [llm-model-optimization](https://github.com/LI-PG1/llm-model-optimization) — 量化基准测试和微调模板
- [llm-rag-agent-service](https://github.com/LI-PG1/llm-rag-agent-service) — RAG + Agent 融合服务
