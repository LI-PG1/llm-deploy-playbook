# llm-deploy-playbook

大模型部署实战笔记。整理了在云平台交付多个模型的过程中遇到的场景对应的方案和踩坑记录。

---

## 目录

* [README.md](README.md)
* [00-前置知识.md](00-%E5%89%8D%E7%BD%AE%E7%9F%A5%E8%AF%86.md) — GPU/显存/精度/分布式术语
* [docs/](docs/)
  * [01-模型选取与下载.md](docs/01-%E6%A8%A1%E5%9E%8B%E9%80%89%E5%8F%96%E4%B8%8E%E4%B8%8B%E8%BD%BD.md) — 选模型、获取与校验
  * [02-确认依赖与版本.md](docs/02-%E7%A1%AE%E8%AE%A4%E4%BE%9D%E8%B5%96%E4%B8%8E%E7%89%88%E6%9C%AC.md) — transformers/CUDA 版本匹配
  * [03-编写Dockerfile.md](docs/03-%E7%BC%96%E5%86%99Dockerfile.md) — 基础镜像、多阶段构建
  * [04-模型量化.md](docs/04-%E6%A8%A1%E5%9E%8B%E9%87%8F%E5%8C%96.md) — BnB/GPTQ/AWQ/FP8
  * [05-vLLM部署.md](docs/05-vLLM%E9%83%A8%E7%BD%B2.md) — 推理引擎部署与调优
  * [06-Gradio界面.md](docs/06-Gradio%E9%9D%A2%E7%95%8C%E6%96%87%E6%A1%A3.md) — WebUI、保活机制
  * [07-MoE模型.md](docs/07-MoE%E6%A8%A1%E5%9E%8B.md) — 专家并行、量化特殊处理
  * [08-代码适配.md](docs/08-%E4%BB%A3%E7%A0%81%E9%80%82%E9%85%8D.md) — 设备/tokenizer/generate
  * [09-模型评测.md](docs/09-%E6%A8%A1%E5%9E%8B%E8%AF%84%E6%B5%8B.md) — TTFT/吞吐/延迟
  * [10-常见错误排查.md](docs/10-%E5%B8%B8%E8%A7%81%E9%94%99%E8%AF%AF%E6%8E%92%E6%9F%A5.md) — 按问题分类速查
  * [11-文件完整性校验.md](docs/11-%E6%96%87%E4%BB%B6%E5%AE%8C%E6%95%B4%E6%80%A7%E6%A0%A1%E9%AA%8C.md) — sha256/断点续传
  * [12-ComfyUI部署.md](docs/12-ComfyUI%E9%83%A8%E7%BD%B2.md) — 扩散模型部署
* [examples/](examples/) — 各类型模型的完整部署示例（共 27 个项目）
