# 07 · MoE 模型

标准稠密（Dense）模型照 04–06 的标准流程即可部署；本篇聚焦 Mixture-of-Experts（MoE）架构——它"总参数大、激活参数小"，因此在显存估算、并行策略与量化上都有专属考量。

## MoE 原理精简版

MoE（Mixture of Experts）的核心思想：**参数多但计算不贵**。

| 概念             | 解释                                     | 类比         |
| :------------- | :------------------------------------- | :--------- |
| **总参数量**       | 所有专家权重的总和，决定模型知识容量                     | 公司总员工数     |
| **激活参数量**      | 每次推理实际参与计算的参数，决定计算成本                   | 当前项目参与人数   |
| **Router（路由）** | 每层的小网络，判断当前 token 交给哪些专家               | 项目经理分配任务   |
| **Top-K**      | 每次激活的专家数，通常 K=2                        | 每个任务派 2 个人 |
| **共享专家**       | 部分 MoE（如 DeepSeek）有一个所有 token 都经过的共享专家 | 公司前台谁都经过   |

典型 MoE 模型（已部署验证）的参数比例：

| 模型                      |   总参数  |   激活参数  |      激活比例      |
| :---------------------- | :----: | :-----: | :------------: |
| DeepSeek-R1-Distill-14B |  \~14B |  \~14B  | 100%（Dense蒸馏版） |
| 某 35B MoE 模型            |  \~35B |   \~3B  |      \~9%      |
| 某 122B MoE 模型           | \~122B |  \~10B  |      \~8%      |
| Keye-VL-2.0-30B-A3B     |  \~30B |   \~3B  |      \~10%     |
| 某 33B MoE 模型            |  \~33B | \~16.5B |      \~50%     |
| 某 26B 扩散 MoE 模型         |  \~26B |   \~4B  |      \~15%     |

> 可以看到 MoE 模型的核心价值：**用 30-122B 的知识容量，只付出 3-16B 的计算成本**。

## 四种并行策略对比

MoE 模型部署到多 GPU 时有四种并行策略可选，理解它们的区别直接影响部署方案的选型：

| 策略                         | 原理                             | 显存节省 |       通信开销      | 适用场景              |
| :------------------------- | :----------------------------- | :--: | :-------------: | :---------------- |
| **Tensor Parallel (TP)**   | 每层权重切分到多卡，每卡算一部分               | ∼1/N | 高（每层 AllReduce） | 2-8 卡，高带宽（NVLink） |
| **Pipeline Parallel (PP)** | 不同层分到不同卡，串行计算                  | ∼1/N |     低（仅层边界）     | 层数 > 64，跨节点       |
| **Data Parallel (DP)**     | 每卡持有完整模型，处理不同请求                |  不省  |     低（仅同步梯度）    | 高并发场景             |
| **Expert Parallel (EP)**   | 不同专家分布到不同卡，Router 做 All-to-All | ∼1/N |  中（All-to-All）  | **MoE 专属**        |

**对 MoE 的建议优先级**：EP > TP > DP > PP

> EP 是最契合 MoE 架构的并行方式 —— 专家本来就是独立的子网络，天然适合分布到不同 GPU 上。其代价是 Router 需要跨卡做 All-to-All 通信来分发/收集 token，因此要求卡间带宽充足（NVLink 或高带宽互联）。

### 各种并行策略的通信模式

```
TP：GPU0 ──AllReduce── GPU1    每层一次全局通信
    │                       │
    W=[A|B]                 W=[A|B]
    （每卡持一半权重）       （每卡持一半权重）

EP：GPU0 ──All-to-All── GPU1    每 MoE 层一次
    │                       │
    Expert[1-4]             Expert[5-8]
    （每卡持不同专家）       （每卡持不同专家）

PP：GPU0 → layers[1-10] → GPU1 → layers[11-20]    单向传递
```

## MoE 模型选型与硬件匹配

已部署的若干 MoE 模型的实际部署规格及测试：

| 模型                      | 总参数 |  激活参数 | 实际使用 GPU              |         推理速度         | 关键注意点                                  |
| :---------------------- | :-: | :---: | :-------------------- | :------------------: | :------------------------------------- |
| 某 35B MoE 模型            | 35B |   3B  | 40GB 单卡规格             |       ∼10 tok/s      | FP8 反量化到 BF16 → 翻倍到 35GB，需 CPU offload |
| 某 35B MoE 模型（同架构）      | 35B |   3B  | 待定                    |          待定          | 与前一同架构                                 |
| Keye-VL-2.0-30B-A3B     | 30B |   3B  | 双卡规格                  |          待测          | VL + MoE 双复杂度，flash\_attn 预装很关键        |
| 某 33B MoE 模型            | 33B | 16.5B | 双卡规格                  |       ∼16 tok/s      | 50% 激活率，显存需求接近同尺寸 Dense 模型             |
| 某 26B 扩散 MoE 模型         | 26B |   4B  | 51.6GB 单卡规格           | 129 tok/s / 2.3s P50 | 扩散+MoE 架构特殊                            |

> 🔧 **选规格前先确认是否支持多卡**：有些 GPU 规格只有单卡，大模型需要多卡 tensor parallelism 时必须选支持多卡的规格。某 35B 模型 FP8 反量化后在原有算力卡上必须 CPU offload，最初选型时没确认算力导致性能未达预期。

### 快速估算公式

```
MoE 模型最少需要显存 ≈ 总参数 × 每参数字节 × 0.5（约一半专家在显存）
推荐显存 ≈ 总参数 × 每参数字节 × 1.2（全部专家 + KV Cache）
```

例：某 122B MoE 模型 BF16：

- 最少：122B × 2 bytes × 0.5 = 122 GB → 至少 2 卡 A100
- 推荐：122B × 2 bytes × 1.2 = 293 GB → 4 卡 H100

## vLLM 部署

vLLM 对 MoE 模型的支持比原生 Transformers 好，能自动处理专家分配。

### 基本用法

```python
from vllm import LLM

llm = LLM(
    model='moe-model-path',
    tensor_parallel_size=2,          # 多卡分配专家
    expert_parallel_size=2,          # 显式指定专家并行（vLLM 0.8+）
    gpu_memory_utilization=0.85,
    max_model_len=8192,
    trust_remote_code=True           # MoE 模型通常需要
)
```

### vLLM 专家并行（Expert Parallelism）

vLLM 0.8+ 开始原生支持 EP。核心参数 `expert_parallel_size`：

| 参数                     | 说明                         | 默认值   |
| :--------------------- | :------------------------- | :---- |
| `tensor_parallel_size` | 张量并行数                      | 1     |
| `expert_parallel_size` | 专家并行数，MoE 专用               | 等于 TP |
| `ep_use_all_gather`    | 使用 AllGather 代替 All-to-All | False |

```
示例：4 卡配置

TP=2, EP=2（推荐）
  每张卡：持有所有专家的一半权重
  GPU0: Expert[1-4]_half + GPU1: Expert[1-4]_half
  GPU2: Expert[5-8]_half + GPU3: Expert[5-8]_half

TP=4, EP=1（纯 TP）
  每张卡：持所有专家的四分之一权重
  通信频繁，但 NVLink 带宽够时吞吐最高

TP=1, EP=4（纯 EP）
  每张卡：持不同专家（完整权重）
  GPU0: Expert[1-2], GPU1: Expert[3-4], GPU2: Expert[5-6], GPU3: Expert[7-8]
```

> 🔧 部署时使用的双卡 A100 适合 TP=2。如果模型所有专家能放进单卡，TP=2 不如单卡；只有单卡放不下全部专家时才需要多卡。

### EPLB（Expert Parallel Load Balancer）

vLLM 内置的负载均衡器，处理专家分布不均的问题：

```bash
# 启用 EPLB
--enable-ep-lb
# 配置负载均衡参数
--ep-lb-config '{"type": "greedy", "max_skew": 0.1}'
```

## device\_map 详解

MoE 模型的 device\_map 选择直接决定部署成败。

### 为什么 MoE 必须用 device\_map='auto'

MoE 模型的专家层权重远超注意力和 embedding 层。`device_map='auto'` 让 accelerate 自动分析每层大小，分配到最合适的设备上。

**不用 device\_map 的后果**：所有权重默认放在 CPU，deepcopy 在 CPU 上遍历数十 B 权重 → 永久卡死。

### 为什么 device\_map='auto' 也可能 OOM

accelerate 的分配策略是"从大到小"——先填满 GPU0，剩下放 GPU1，再剩下放 CPU。对 MoE 模型，它可能在 GPU0 上放了多个完整专家，导致显存溢出。

**三种配置示例**：

```python
# 配置 1：单卡 + CPU offload（推荐起步）
model = AutoModelForCausalLM.from_pretrained(
    'model-path',
    device_map='auto',
    max_memory={0: '20GB', 'cpu': '40GB'},   # GPU 留 4GB 给 KV Cache
    torch_dtype=torch.bfloat16
)

# 配置 2：双卡
model = AutoModelForCausalLM.from_pretrained(
    'model-path',
    device_map='auto',
    max_memory={0: '24GB', 1: '24GB', 'cpu': '40GB'},
    torch_dtype=torch.bfloat16
)

# 配置 3：手动指定设备
from transformers import AutoConfig
config = AutoConfig.from_pretrained('model-path')
num_experts = config.num_local_experts  # 拿到专家数
# 然后手动映射每层到指定 GPU
```

> 🔧 部署时遇到的实际问题：某 35B 模型 FP8 反量化到 BF16 后 54% OOM，最终用 `max_memory={0: '20GB', 'cpu': '60GB'}` 才加载成功，推理速度 ∼10 tok/s。

## flash\_attn 预装策略

非标准 MoE/VL 架构（如 Keye-VL-2、某扩散 MoE）的 modeling 代码可能硬 import `flash_attn`。在部署平台上现场编译 flash\_attn 几乎不可行（32GB RAM 不足 + 相关源码托管站点被墙）。

**推荐策略**：

```dockerfile
# 选预装 flash_attn 的基础镜像
# 4.40 前可用
FROM pytorch2.4-flash_attn2.6.3-cuda11.8
# 4.48+ 后
FROM pytorch2.5-flash_attn3.0.0-cuda12.4
```

**先查模型是否需要**：

```bash
# 在模型目录里搜 import flash_attn
grep -r "flash_attn" model_path/
# 如果有结果 → 用预装 flash_attn 的镜像
```

> 🔧 例：Keye-VL-2 的 modeling 代码硬 import `flash_attn.layers.rotary`，选了预装镜像后加载时间从 2h+→40min。

## MoE 量化考虑

MoE 模型的量化与 Dense 模型有一些重要差异：

| 量化方案             | MoE 兼容性 | 说明                           |
| :--------------- | :-----: | :--------------------------- |
| bitsandbytes NF4 |   ✅ 可用  | 加载时即时量化，但专家层量化/解量化开销大        |
| GPTQ             |  ⚠️ 需注意 | 校准数据要覆盖专家的多样性，否则量化后质量波动大     |
| AWQ              |   ✅ 推荐  | 激活感知量化对 MoE 的 Router 影响最小    |
| FP8（原生）          |  ✅ 最适合  | H100/L40S 原生支持，**MoE 的理想精度** |

**FP8 的核心优势**：MoE 的总参数大但激活参数小，用 FP8 可以把所有专家放进显存同时保持激活参数的计算精度。

> 🔧 某 35B 模型官方 FP8 版本在 40GB 单卡规格（算力 8.0）上反量化到 BF16，35B 总参数从 ∼17.5GB 膨胀到 ∼35GB。如果 GPU 原生支持 FP8（L40S 算力 8.9），不会触发反量化，推理速度会快很多。

## 负载均衡与利用率

### 专家偏斜问题

MoE 的 Router 不是均匀分配 token 的——部分"热门专家"可能收到远超平均的 token 量：

```
理想：Expert1(25%) Expert2(25%) Expert3(25%) Expert4(25%)
实际：Expert1(47%) Expert2(31%) Expert3(15%) Expert4(7%)
```

这种偏斜导致：

- 热门专家的 GPU 利用率高，冷门专家闲置
- 总体 GPU 利用率可能 < 30%
- 部分请求因等待热门专家而延迟

### EPLB 的原理

EPLB（Expert Parallel Load Balancer）通过**复制热门专家**到多张 GPU 来解决偏斜：

```
没有 EPLB：                     有 EPLB：
GPU0: Expert[1,2,3,4]           GPU0: Expert[1,2,3]  ← Expert2 副本
GPU1: Expert[5,6,7,8]           GPU1: Expert[4,5,6,7,8]  ← Expert2 原始
                                如果 Expert2 最热，两张卡都能处理它
```

vLLM 0.8+ 中通过 `--enable-ep-lb` 启用。

## 常见错误

| 错误                                                    | 原因                | 方案                                 |
| :---------------------------------------------------- | :---------------- | :--------------------------------- |
| CUDA OOM 在 54%                                        | `device_map` 分配不均 | 加 `max_memory` + CPU offload       |
| CPU loading 1 小时到 4%                                  | 没加 `device_map`   | 加 `device_map='auto'`              |
| FP8 反量化到 BF16 体积翻倍                                    | GPU 算力 < 8.9      | 直接下 BF16 版本                        |
| transformers import 报错 `KeyError: 'qwen3_5_moe'`      | transformers 版本过旧 | 升级到 5.x + `trust_remote_code=True` |
| `ModuleNotFoundError: transformers_modules.Keye-VL-2` | 模型目录含连字符          | 升级 transformers 4.57.0+，或手动改名      |
| Flash Attention 编译失败                                  | 部署平台 32GB RAM 不足 | 选预装 flash\_attn 的镜像                |
| hf-mirror 下载到乱码权重                                     | 门控模型无 token       | hf-mirror + HF token 逐文件下载         |

## 多卡容器配置

多 GPU 场景需要 `torchrun` 启动：

```bash
# 单节点多卡
torchrun --nproc_per_node=2 --nnodes=1 app.py

# 不同框架对 tp_size 的支持
# vLLM: tensor_parallel_size=2
# Transformers: device_map='auto'（自动用满）
# deepspeed: --num_gpus=2
```

### NCCL 调试

多卡通信失败时，设置以下环境变量查看详细日志：

```bash
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
export NCCL_IB_DISABLE=1          # 无 InfiniBand 时禁用
export NCCL_SOCKET_IFNAME=eth0    # 指定网卡
export NCCL_P2P_DISABLE=1         # P2P 失败时禁用
```

> 🔧 例：某 122B 模型在双卡上遇到 RPC 连接拒绝，最终发现是防火墙阻塞了卡间通信。设置 `NCCL_SOCKET_IFNAME=eth0` 后解决。

## 多卡通信 FAQ

| 问题                        | 原因          | 方案                                        |
| :------------------------ | :---------- | :---------------------------------------- |
| `RPC: connect refused`    | 防火墙/路由阻塞    | 指定 `NCCL_SOCKET_IFNAME`，或单卡 + CPU offload |
| `AllReduce timed out`     | NVLink 带宽不足 | 降 `tensor_parallel_size`，或换用 EP           |
| `CUDA error: peer access` | P2P 不可用     | 设 `NCCL_P2P_DISABLE=1`                    |
| 显存分配严重不均                  | 专家偏斜        | 启用 EPLB，或手动分配 expert->GPU                 |
| 多卡不如单卡快                   | 通信开销 > 计算收益 | 先测单卡能不能放下，放不下再开多卡                         |
