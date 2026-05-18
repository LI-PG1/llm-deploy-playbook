# 05 · vLLM 部署

量化方案确定后，用 vLLM 将模型封装为高吞吐推理服务。

> **vLLM 是什么**：一个专注于大模型推理吞吐的 serving 框架。其核心是 PagedAttention——像操作系统管理内存分页一样管理 KV Cache，避免显存碎片；配合连续批处理（continuous batching），能把多条并发请求拼到同一批计算里。相比原生 `transformers` 推理，vLLM 在并发场景下通常带来 2–4 倍吞吐提升。当一个模型需要对外提供稳定、高并发的服务时，优先选 vLLM。

## 基本用法

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model='model-path',
    tensor_parallel_size=1,    # GPU 数量
    gpu_memory_utilization=0.9 # 最大显存使用率
)
params = SamplingParams(temperature=0.7, max_tokens=1024)
outputs = llm.generate(['prompt'], params)
```

## 常用参数

| 参数                           | 作用                  |
| :--------------------------- | :------------------ |
| `tensor_parallel_size=N`     | 将模型分到 N 张 GPU       |
| `gpu_memory_utilization=0.9` | 最大可用显存比例            |
| `max_model_len=32768`        | 上下文窗口上限             |
| `enforce_eager=True`         | 禁用 CUDA graph（调试模式） |

## Docker 部署

```dockerfile
FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

RUN pip install vllm==0.8.3

# 用 --no-deps 避免子依赖触发外网下载
# RUN pip install vllm==0.8.3 --no-deps && pip install einops ray

COPY app.py /app/
WORKDIR /app
CMD ["python", "app.py"]
```

> 🔧 vLLM 的 overlay tar.gz 源包在编译时可能触发外网下载，网络不通时永久卡死（CPU=0%）。推荐从 wheel 文件安装或用 `--no-deps` 后单独补缺的包。

## OpenAI 兼容 API

vLLM 提供了兼容 OpenAI API 的服务器模式：

```bash
# 启动 API 服务器
python -m vllm.entrypoints.openai.api_server \
    --model /path/to/model \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 8192 \
    --port 8000
```

```python
# 客户端调用（兼容 OpenAI SDK）
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"  # vLLM 不验证 key
)

response = client.chat.completions.create(
    model="model-name",
    messages=[{"role": "user", "content": "Hello"}],
    temperature=0.7,
    max_tokens=512
)
```

> 兼容 OpenAI API 的好处：可以无缝接入现有工具链（LangChain、AutoGen、Open WebUI 等），无需修改客户端代码。

## 性能调优

### 核心参数

| 参数 | 建议 | 说明 |
|:----|:----|:-----|
| `gpu_memory_utilization=0.9` | 0.85-0.95 | 预留 KV Cache + 系统开销 |
| `max_num_seqs=256` | 并发请求上限 | 太大增加 OOM 风险 |
| `max_num_batched_tokens=8192` | 单批最大 token 数 | 控制批处理时效 |
| `enable_prefix_caching=True` | 共享前缀缓存 | 重复请求场景显著加速 |

### 吞吐对标

| 引擎 | TPS 提升 | 显存节省 | 适用场景 |
|:----|:--------:|:--------:|:--------|
| Transformers 原生 | 基准 (1.0x) | 基准 | 通用，配置最少 |
| vLLM（单卡） | 2-4x | ~20% | 高并发生产 |
| TGI | 1.5-3x | ~15% | HF 生态用户 |
| TensorRT-LLM | 3-6x | ~30% | 极致优化，H100 最佳 |

> 🔧 性能测试时建议统一推理引擎再对比精度/量化方案的影响，不要同时变两个变量（如「vLLM + INT4 vs Transformers + BF16」无法归因）。

## 常见错误

### overlay 编译永久卡死

安装 vLLM overlay（tar.gz 源包）时，`setup.py` 编译可能触发外网下载，如果网络不通会永久卡死（CPU=0%）。

**方案**：

- 从 wheel 文件安装
- 确认安装时网络可达
- 用 `--no-deps` 避免子依赖触发外网下载

### OOM（显存不足）

```python
# 降低显存压力
llm = LLM(
    model='model-path',
    gpu_memory_utilization=0.7,  # 从 0.9 降至 0.7
    max_model_len=16384,          # 缩短上下文
    tensor_parallel_size=2        # 用 2 张 GPU
)
```

### 部署实例：Transformers 5.x + Qwen2 架构 tokenizer 问题

vLLM 内部调用 `tokenizer.all_special_tokens_extended`，该属性在 transformers 5.x 的 Qwen2Tokenizer 中被移除。

**方案**：在 import vLLM 前加 monkey-patch

```python
from transformers.models.qwen2.tokenization_qwen2 import Qwen2Tokenizer
if not hasattr(Qwen2Tokenizer, 'all_special_tokens_extended'):
    Qwen2Tokenizer.all_special_tokens_extended = property(lambda self: [])
```

### 注意力 N\^2 实现不支持

当 vLLM 报错关于 attention backend 不支持时，通常是因为模型架构不在 vLLM 的预定义支持列表中：

```bash
# 切换到 PLAIN 注意力实现（不依赖 flash_attn）
VLLM_ATTENTION_BACKEND=FLASH_ATTN  # 默认
VLLM_ATTENTION_BACKEND=PLAIN       # 降级选项
```

> 🔧 非标准注意力架构（如某些扩散 MoE 模型）在 vLLM 中无法推理。先确认模型在 vLLM 支持的模型列表中，否则直接用 Transformers。

## 调试技巧

| 场景        | 做法                                            |
| :-------- | :-------------------------------------------- |
| vLLM 加载报错 | 先换 `transformers` 推理确认模型文件正常                  |
| 需要排查底层问题  | `enforce_eager=True` 关闭 CUDA graph            |
| 显存不够      | 减少 `gpu_memory_utilization` 或 `max_model_len` |
| 多卡分布不均    | 调整 `tensor_parallel_size`                     |
| 模型加载隔离测试 | `--load-format dummy` 不加载权重，只测流程             |
| 通信卡死      | `NCCL_DEBUG=TRACE` 打开 NCCL 日志排查               |
