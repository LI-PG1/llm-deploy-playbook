# 06 · Gradio 界面

模型已能对外提供推理服务，但要让人直接对话与调试，还需要一个交互界面——下面介绍如何基于 Gradio 快速搭建。

> **Gradio 是什么**：一个为机器学习模型快速搭建 Web UI 的 Python 库。它把输入控件（文本框、图片上传等）和模型推理函数绑定，几行代码即可生成带本地/局域网访问地址的页面，常被用作推理服务的轻量前端或内部演示界面。

## 基础聊天界面

```python
import gradio as gr

def chat(message, history):
    # 调用模型生成答案
    response = model.generate(message)
    return response

gr.ChatInterface(
    fn=chat,
    title='聊天',
    description='与模型对话'
).launch(server_name='0.0.0.0', server_port=7860)
```

## 常用模式

| 模式 | 适用场景 |
|:----|:--------|
| `ChatInterface` | 简单对话，单轮或多轮 |
| `Blocks` | 复杂界面（多输入、多输出、多 Tab） |
| `gr.State` | 保持对话历史、模型引用 |

## Gradio 5.x/6.x 变化

### 消息格式从 tuple 改为 dict

```python
# 旧版 Gradio 4.x（返回 tuples）
# return "回答"

# Gradio 5.x/6.x（返回 dict 列表）
return [{"role": "user", "content": message},
        {"role": "assistant", "content": response}]
```

> `gr.Chatbot(type="tuples")` 已经不再支持，不用加 `type` 参数。

## 常见问题

| 问题 | 方案 |
|:----|:-----|
| 端口被占用 | 换 `server_port` 或杀掉旧进程 |
| 每次请求都重载模型 | 模型加载放函数外面，用 global 或 `gr.State` 引用 |
| `max_new_tokens` 不生效 | 在 `model.generate()` 里显式传参 |
| 流式输出不工作 | 设 `stream=True`，chat 函数改为 generator |
| 日志一堆 WARNING | 部署平台健康检查的正常输出，不影响功能，可压制 |

## 后台加载大模型

当模型加载时间很长（>1h）时，需要在前台启动 Gradio 后后台加载模型：

```python
import threading
from threading import Event

MODEL_LOADED = Event()

def load_model():
    # 复杂加载逻辑
    global model
    model = ...
    MODEL_LOADED.set()

# 先启动 Gradio（前台），再后台加载模型
threading.Thread(target=load_model, daemon=True).start()

# 保活线程 — 每 5 分钟发 HTTP 请求防止空闲超时
def keepalive():
    while not MODEL_LOADED.wait(timeout=300):
        try:
            urllib.request.urlopen('http://127.0.0.1:7860')
        except:
            pass
```

## 日志与调试

```python
import logging
# 压制健康检查告警
logging.getLogger('uvicorn.error').disabled = True
```
