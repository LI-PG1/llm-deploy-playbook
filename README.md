# llm-deploy-playbook

大模型部署实战笔记。包含多个模型从容器化到推理服务化的完整示例。

## docs/ — 部署文档

- [00-前置知识.md](./00-前置知识.md) — GPU/显存/精度/分布式术语
- [01-模型选取与下载.md](./docs/01-模型选取与下载.md) — 选模型、下载与校验
- [02-确认依赖与版本.md](./docs/02-确认依赖与版本.md) — transformers/CUDA 版本匹配
- [03-编写Dockerfile.md](./docs/03-编写Dockerfile.md) — 基础镜像、多阶段构建
- [04-模型量化.md](./docs/04-模型量化.md) — BnB/GPTQ/AWQ/FP8 方案对比
- [05-vLLM部署.md](./docs/05-vLLM部署.md) — 推理引擎部署与参数调优
- [06-Gradio界面.md](./docs/06-Gradio界面.md) — WebUI、保活机制
- [07-MoE模型.md](./docs/07-MoE模型.md) — 专家并行、EPLB、量化
- [08-代码适配.md](./docs/08-代码适配.md) — 设备/tokenizer/generate 适配
- [09-模型评测.md](./docs/09-模型评测.md) — TTFT/吞吐/P50-P95 评测方法
- [10-常见错误排查.md](./docs/10-常见错误排查.md) — 按问题分类速查
- [11-文件完整性校验.md](./docs/11-文件完整性校验.md) — sha256/断点续传
- [12-ComfyUI部署.md](./docs/12-ComfyUI部署.md) — 扩散模型部署

## examples/ — 部署示例

27 个已部署模型的完整工程文件，按场景分类：

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

## 相关仓库

- [llm-model-optimization](https://github.com/LI-PG1/llm-model-optimization)
- [llm-rag-agent-service](https://github.com/LI-PG1/llm-rag-agent-service)
