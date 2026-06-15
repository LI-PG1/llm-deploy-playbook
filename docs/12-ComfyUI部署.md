# 12 · ComfyUI 部署

本文是专项之一：扩散类模型（SD / Flux / LTX Video 等）通过 ComfyUI API 模式部署。与对话模型的 Gradio 直连不同，这类模型依赖 ComfyUI 的节点化工作流与队列，部署要点集中在路径配置、workflow 转换与生产架构。

## 判断是否需要 ComfyUI

| 模型类型 | 推荐方案 |
|:--------|:--------|
| SD / Flux / LTX Video 等扩散模型 | ComfyUI API 模式 |
| TTS / LLM 等 | Gradio 直连 |
| 模型架构复杂、有自定义节点 | ComfyUI |

## 前置条件

- torch 升级时用 `--no-deps` 避免 NCCL 锁
- 不要用 conda 装 torch（可能缺 intel-itt 库）
- ComfyUI 0.22+ 需要 `comfyui-frontend-package` 作为硬依赖
- 自定义节点从 GitHub 下载 zip 上传，不在容器内下载
- **`av` 包在部分内网源不存在**，需从清华源装：`pip install av comfyui-frontend-package --index-url https://pypi.tuna.tsinghua.edu.cn/simple`
- **部分前端包在内网源版本滞后**（comfyui-frontend-package 最高 1.17.5），版本不够时可考虑用较新基础镜像
- 🔧 **不要依赖容器内下载任何东西** — JSON 文件、节点 zip、模型权重全部在容器外下载好后上传。容器内相关源码站点被墙，HF 不可达

## ComfyUI API 架构

ComfyUI 提供 5 个核心 API 端点：

| 端点 | 方法 | 用途 |
|:----|:----|:-----|
| `/ws` | WebSocket | 实时推送任务进度和日志 |
| `/prompt` | POST | 提交工作流到执行队列 |
| `/history/{prompt_id}` | GET | 查询任务完成后的结果 |
| `/view` | GET | 下载生成的图片 |
| `/upload/{image_type}` | POST | 上传输入图片（如 input/inpaint） |
| `/workflow/convert` | POST | 将完整 workflow 转换为 API 格式（需安装自定义节点） |

**典型请求-响应流程**：

```
Client                     ComfyUI Server
  │                             │
  │  POST /upload (input.png)   │
  │────────────────────────────>│  返回 image_name
  │                             │
  │  POST /prompt (workflow)    │
  │────────────────────────────>│  返回 prompt_id
  │                             │
  │  WebSocket /ws              │
  │════════════════════════════>│  实时推送进度 %
  │                             │
  │  GET /history/{prompt_id}   │
  │────────────────────────────>│  返回输出文件列表
  │                             │
  │  GET /view?filename=...     │
  │────────────────────────────>│  下载生成图片
```

## 模型路径配置

```yaml
# extra_model_paths.yaml
comfyui:
    base_path: /opt/ComfyUI/
    checkpoints: /path/to/model/
    vae: /path/to/model/
    clip: /path/to/model/
```

> checkpoint 名称必须用实际文件名，VAE 名称不匹配时改为 `"pixel_space"`。

模型文件路径的注意事项：
- 通过 `extra_model_paths.yaml` 或在 `start.sh` 中用 `ln -s` 软链接指向实际挂载路径
- 不能假设模型在 ComfyUI 默认路径（`models/checkpoints/`）
- 路径必须与实际挂载路径**完全一致**

## workflow 转 API payload

ComfyUI GUI 的 workflow 格式不能直接用作 API payload，需要转换。

### 两种格式的区别

| 格式 | 来源 | 包含 Reroute/Note 等 GUI 节点 | 能否提交到 /prompt |
|:----|:----|:---------------------------|:-----------------|
| **完整 workflow** | File → Save（默认） | ✅ | ❌ 会报错 |
| **API workflow** | File → Export (API) | ❌ | ✅ |
| **转换后 API** | 通过 `/workflow/convert` 端点 | ❌ | ✅ |

> **陷阱**：使用完整 workflow 时，如果拖入 API workflow 编辑，再导出会丢失节点布局和部分 widget 值。建议始终保留完整 workflow 版本，通过工具或端点转换后用于 API。

### 转换要点

1. 建立节点索引，追踪 Reroute 递归链
2. 跳过 GUI-only 节点（Reroute / Note / Primitive）
3. 按 links 填入连接关系
4. 通过 `/object_info` API 获取每个节点 required inputs 顺序
5. widget_values 只填充未被 link 覆盖的 required input

> ⚠️ **widget_values 和 inputs 不是简单 index 对应**。必须调用 `/object_info` 获取每个节点 required input 的顺序，筛掉已被 link 填充的输入，再按序映射 widget_values。否则某些节点（如 SaveVideo）会缺 codec/format/filename_prefix 等参数。

## 生产环境部署架构

对于需要稳定服务的场景，推荐以下架构：

```
                    ┌──────────────┐
                    │   Nginx LB   │
                    │   (可选)     │
                    └──────┬───────┘
                           │
               ┌───────────┴───────────┐
               │                       │
        ┌──────┴──────┐        ┌──────┴──────┐
        │ ComfyUI     │        │ ComfyUI     │
        │ Instance 1  │        │ Instance 2  │
        │ GPU: T4     │        │ GPU: T4     │
        └──────┬──────┘        └──────┬──────┘
               │                      │
        ┌──────┴──────┐        ┌──────┴──────┐
        │ 外部存储    │        │ 外部存储    │
        │ (OSS/S3)    │        │ (OSS/S3)    │
        └─────────────┘        └─────────────┘
```

| 组件 | 建议 |
|:----|:----|
| 负载均衡 | Nginx / 云平台 LB |
| 外部存储 | 生成结果存外部存储，不占容器内磁盘 |
| 多实例 | 根据并发需求部署 N 个 ComfyUI 实例 |
| 队列 | 自行实现任务队列（Redis + Celery）避免请求积压 |

## Gitee 不稳定时的替代方案

ComfyUI 或自定义节点从 Gitee 克隆超时时：

1. 从 GitHub 下载 zip 包（提前在本地下载好）
2. 将 zip 上传到项目代码目录
3. 在 Dockerfile 中用 `COPY` 放入镜像，或 start.sh 中解压到目标路径

## 已验证的开发环境安装步骤

以下步骤在某视频生成部署项目中验证通过：

```bash
# 1. 升级 torch（--no-deps 避免 NCCL 锁）
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121 --no-deps

# 2. 补 cuDNN
pip install nvidia-cudnn-cu12==9.1.0.70 --index-url https://download.pytorch.org/whl/cu121

# 3. 降级不兼容包
pip install --force-reinstall numpy==1.26.4 huggingface-hub==0.25.2 transformers==4.40.0

# 4. 克隆 ComfyUI（或上传 zip 解压）
cd /opt && git clone --depth 1 https://gitee.com/mirrors/ComfyUI.git

# 5. 装 ComfyUI 依赖
cd /opt/ComfyUI && pip install einops safetensors aiohttp yarl pyyaml Pillow scipy tqdm psutil alembic SQLAlchemy filelock requests comfy-aimdo==0.3.0 --index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip install av comfyui-frontend-package --index-url https://pypi.tuna.tsinghua.edu.cn/simple

# 6. 解压自定义节点 zip → /opt/ComfyUI/custom_nodes/ → mv 去 -main 后缀

# 7. 装节点依赖
pip install kornia ninja simpleeval "transformers>=4.47" --index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip install --force-reinstall numpy==1.26.4 scipy --index-url https://pypi.tuna.tsinghua.edu.cn/simple

# 8. 配置模型路径
cat > /opt/ComfyUI/extra_model_paths.yaml << 'EOF'
comfyui:
    base_path: /opt/ComfyUI/
    checkpoints: /path/to/model/
    vae: /path/to/model/
    clip: /path/to/model/
EOF

# 9. 启动
export LD_LIBRARY_PATH=/root/miniconda3/lib:$LD_LIBRARY_PATH
cd /opt/ComfyUI && python main.py --listen 0.0.0.0 --port 8188 --disable-auto-launch --output-directory /tmp/output
```

## LTX Video 工作流的自定义节点清单

| 节点 | 来源 |
|:----|:-----|
| **ComfyUI-LTXVideo** | GitHub 下载 zip 上传 |
| **ComfyMath** | GitHub 下载 zip 上传 |
| **ComfyUI-KJNodes** | GitHub 下载 zip 上传 |

以上三个都是从 GitHub 下载 zip 上传，不要在容器内 git clone。

## 常见问题

| 问题 | 原因 | 方案 |
|:----|:----|:-----|
| `Node 'Reroute' not found` | Reroute 是 GUI 节点，API 模式不存在 | 递归追踪 link 链，将输入重定向到真实源节点 |
| `value_not_in_list` | checkpoint 名称不匹配 | 用实际文件名 |
| `av` 包找不到 | 内网源没有这个包 | 从清华源单独装 |
| VAE 不存在 | 名称不对 | 用 ComfyUI 内置选项 `"pixel_space"` |
| 前端包缺失导致退出 | `comfyui-frontend-package` 版本满足要求 | 从对应源安装 |
| `undefined symbol: iJIT_NotifyEvent` | conda 装的 torch 缺 intel-itt 库 | 换 pip 装 torch，不用 conda |
| 上传的 workflow 提交报错 | 不是 API 格式 | 先用 `Export (API)` 导出或用 `/workflow/convert` 转换 |
| WebSocket 连接断开 | 长时间无活动或实例重启 | 客户端实现断线重连机制 |
