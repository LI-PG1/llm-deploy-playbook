# llm-deploy-playbook

大模型部署实战笔记。在趋动云平台交付多个模型的过程中记录的踩坑和解决方案。

---

## 目录结构

```
├── README.md                # 本文件
├── 00-前置知识.md            # GPU/显存/精度/分布式等术语卡
├── docs/
│   ├── 01-模型选取与下载.md    # 选模型、获取与校验
│   ├── 02-确认依赖与版本.md    # transformers/tokenizers/CUDA 版本匹配
│   ├── 03-编写Dockerfile.md   # 基础镜像、多阶段构建、缓存优化
│   ├── 04-模型量化.md         # BnB/GPTQ/AWQ/FP8 方案对比
│   ├── 05-vLLM部署.md        # 推理引擎部署与参数调优
│   ├── 06-Gradio界面.md      # WebUI、多模态、保活机制
│   ├── 07-MoE模型.md         # 专家并行、EPLB、量化特殊处理
│   ├── 08-代码适配.md        # 设备/tokenizer/generate/路径适配
│   ├── 09-模型评测.md        # TTFT/吞吐/P50-P95 评测方法
│   ├── 10-常见错误排查.md     # 按问题分类的速查手册
│   ├── 11-文件完整性校验.md   # sha256/断点续传/校验脚本
│   └── 12-ComfyUI部署.md     # 扩散模型部署
└── examples/
    ├── phi4-mini/            # 4B 模型单卡 FP16 示例
    └── deepseek-r1/          # 14B 多卡 BF16 + CoT 示例
```

## 快速导航

| 你在找什么 | 去这里 |
|-----------|--------|
| 部署流程 | 01 → 02 → 03 → 05 → 06 → 09 |
| MoE 部署 | 07 + 04(量化) + 08(适配) |
| 报错了 | 10(速查手册) |
| 需要跑个例子 | examples/phi4-mini 或 examples/deepseek-r1 |
| Docker 镜像 | 03(写法) + examples(完整文件) |
| 性能调优 | 05(vLLM参数) + 09(评测) |

## 相关仓库

- [llm-model-optimization](https://github.com/LI-PG1/llm-model-optimization) — 量化基准测试和微调模板
- [llm-rag-agent-service](https://github.com/LI-PG1/llm-rag-agent-service) — RAG + Agent 融合服务
