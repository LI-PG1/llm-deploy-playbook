# vLLM 部署

推理服务配置：tensor-parallel-size、max-num-seqs、gpu-memory-utilization。

低并发下 vLLM 约 1.5-2× Transformers，高并发可达 ~3×。