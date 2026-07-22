# llm-deploy-playbook

大模型部署实战笔记。整理了在云平台交付多个模型的过程中遇到的场景对应的方案和踩坑记录。

---

## 目录

* [README.md](./README.md)
* [00-前置知识.md](./00-前置知识.md) — GPU/显存/精度/分布式术语
* [docs/](./docs/)
  * [01-模型选取与下载.md](./docs/01-模型选取与下载.md) — 选模型、获取与校验
  * [02-确认依赖与版本.md](./docs/02-确认依赖与版本.md) — transformers/CUDA 版本匹配
  * [03-编写Dockerfile.md](./docs/03-编写Dockerfile.md) — 基础镜像、多阶段构建
  * [04-模型量化.md](./docs/04-模型量化.md) — BnB/GPTQ/AWQ/FP8
  * [05-vLLM部署.md](./docs/05-vLLM部署.md) — 推理引擎部署与调优
  * [06-Gradio界面.md](./docs/06-Gradio界面.md) — WebUI、保活机制
  * [07-MoE模型.md](./docs/07-MoE模型.md) — 专家并行、量化特殊处理
  * [08-代码适配.md](./docs/08-代码适配.md) — 设备/tokenizer/generate
  * [09-模型评测.md](./docs/09-模型评测.md) — TTFT/吞吐/延迟
  * [10-常见错误排查.md](./docs/10-常见错误排查.md) — 按问题分类速查
  * [11-文件完整性校验.md](./docs/11-文件完整性校验.md) — sha256/断点续传
  * [12-ComfyUI部署.md](./docs/12-ComfyUI部署.md) — 扩散模型部署
* [examples/](./examples/) — 各类型模型的完整部署示例
