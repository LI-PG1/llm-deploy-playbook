# llm-deploy-playbook

大模型部署实战笔记。在趋动云平台交付 21 个模型的过程中记录的踩坑和解决方案。

## 内容

- **Docker 容器化**：基础镜像选型、依赖锁版本、多阶段构建
- **vLLM 推理服务**：TP/EP 配置、continuous batching 调优、MoE 多卡切分
- **模型量化**：bitsandbytes 运行时量化、GPTQ/AWQ 离线量化
- **Gradio/FastAPI 服务化**：保活线程、多模型路由
- **常见错误排查**：从依赖冲突到 NCCL timeout，按问题分类可查

## 相关仓库

- [llm-model-optimization](https://github.com/LI-PG1/llm-model-optimization) — 量化基准测试和微调模板
- [llm-rag-agent-service](https://github.com/LI-PG1/llm-rag-agent-service) — RAG + Agent 融合服务
