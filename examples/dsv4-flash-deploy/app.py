"""
DeepSeek-V4-Flash Gradio Chat Interface
连接到 vLLM API 服务器，支持三种推理模式
"""

import os
import json
import time
from openai import OpenAI

import gradio as gr

# ===== 配置 =====
API_BASE = os.environ.get("VLLM_API_BASE", "http://localhost:8000/v1")
API_KEY = os.environ.get("VLLM_API_KEY", "EMPTY")
MODEL_NAME = os.environ.get("VLLM_MODEL_NAME", "deepseek-v4-flash")

client = OpenAI(base_url=API_BASE, api_key=API_KEY)


# ===== 检查服务器状态 =====
def check_server():
    try:
        models = client.models.list()
        model_ids = [m.id for m in models]
        print(f"[OK] Connected to vLLM server at {API_BASE}")
        print(f"[OK] Available models: {model_ids}")
        return True, model_ids
    except Exception as e:
        print(f"[ERROR] Cannot connect to vLLM server: {e}")
        return False, []


SERVER_OK, AVAILABLE_MODELS = check_server()
if not SERVER_OK:
    print("[WARN] vLLM server not reachable. Start it with: bash start.sh")


# ===== 对话逻辑 =====
class DeepSeekChat:
    def __init__(self):
        self.conversation = []

    def reset(self):
        self.conversation = []
        return []

    def chat(self, user_message, reasoning_mode, temperature, max_tokens, history):
        if not user_message.strip():
            return history

        # 构建消息
        messages = []
        for turn in self.conversation:
            messages.append({"role": "user", "content": turn["user"]})
            messages.append({"role": "assistant", "content": turn["assistant"]})
        messages.append({"role": "user", "content": user_message})

        # 推理模式映射
        reasoning_map = {
            "Non-think（快速回答）": {},
            "Think High（深度思考）": {"thinking": True, "reasoning_effort": "high"},
            "Think Max（最大推理）": {"thinking": True, "reasoning_effort": "max"},
        }
        chat_template_kwargs = reasoning_map.get(reasoning_mode, {})

        # 构建请求
        extra_body = {}
        if chat_template_kwargs:
            extra_body["chat_template_kwargs"] = chat_template_kwargs

        try:
            start_time = time.time()

            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=float(temperature),
                max_tokens=int(max_tokens),
                stream=False,
                extra_body=extra_body if extra_body else None,
            )

            elapsed = time.time() - start_time
            raw_text = response.choices[0].message.content or "(empty response)"

            # 提取思考过程和回答
            display_text = raw_text
            thinking_content = ""
            if "<think>" in raw_text and "</think>" in raw_text:
                import re
                match = re.search(r"<think>(.*?)</think>\s*(.*)", raw_text, re.DOTALL)
                if match:
                    thinking_content = match.group(1).strip()
                    answer = match.group(2).strip()
                    display_text = answer if answer else raw_text

            # 保存对话
            self.conversation.append({
                "user": user_message,
                "assistant": raw_text,
                "thinking": thinking_content,
                "latency": f"{elapsed:.1f}s",
            })

            # 构建 Gradio 6.x 格式
            gradio_messages = []
            for turn in self.conversation:
                content = turn["assistant"]
                if turn.get("thinking"):
                    content = f"**[推理过程]**\n```\n{turn['thinking']}\n```\n\n**[回答]**\n{content}"
                gradio_messages.append({"role": "user", "content": turn["user"]})
                gradio_messages.append({"role": "assistant", "content": content + f"\n\n*({turn.get('latency', '')})*"})

            return gradio_messages

        except Exception as e:
            import traceback
            error_msg = f"Error: {type(e).__name__}: {e}"
            print(error_msg)
            traceback.print_exc()

            self.conversation.append({"user": user_message, "assistant": error_msg})
            gradio_messages = []
            for turn in self.conversation:
                gradio_messages.append({"role": "user", "content": turn["user"]})
                gradio_messages.append({"role": "assistant", "content": turn["assistant"]})
            return gradio_messages


chat_instance = DeepSeekChat()


# ===== Gradio UI =====
TITLE = "DeepSeek-V4-Flash Chat"

DESCRIPTION = """
**DeepSeek-V4-Flash** — 284B MoE (13B active), FP4+FP8 mixed precision, 1M context.

Powered by **vLLM** inference engine.
"""

with gr.Blocks(title=TITLE, theme=gr.themes.Soft()) as demo:
    gr.Markdown(f"# {TITLE}")
    gr.Markdown(DESCRIPTION)

    with gr.Row():
        with gr.Column(scale=1):
            with gr.Accordion("推理模式", open=True):
                reasoning_mode = gr.Radio(
                    choices=[
                        "Non-think（快速回答）",
                        "Think High（深度思考）",
                        "Think Max（最大推理）",
                    ],
                    value="Think High（深度思考）",
                    label="Reasoning Effort",
                    info="Think Max 需 max-model-len ≥ 384K",
                )

            with gr.Accordion("生成参数", open=False):
                temperature = gr.Slider(
                    0.0, 2.0, 1.0, step=0.1,
                    label="Temperature",
                    info="推荐 1.0",
                )
                max_tokens = gr.Slider(
                    256, 16384, 4096, step=256,
                    label="Max Tokens",
                )

            reset_btn = gr.Button("🗑 清除对话", variant="secondary", size="lg")

            if not SERVER_OK:
                gr.Warning(
                    "⚠️ vLLM 服务器未连接。请先在终端运行：bash start.sh",
                    visible=True,
                )

        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                label="对话",
                height=600,
                type="messages",
                show_copy_button=True,
            )

            with gr.Row():
                msg_input = gr.Textbox(
                    label="输入消息",
                    placeholder="输入你的问题，按 Enter 发送...",
                    lines=2,
                    scale=10,
                )
                send_btn = gr.Button("发送", variant="primary", size="lg", scale=1)

    # 绑定事件
    send_btn.click(
        fn=chat_instance.chat,
        inputs=[msg_input, reasoning_mode, temperature, max_tokens, chatbot],
        outputs=[chatbot],
    ).then(lambda: "", None, msg_input)

    msg_input.submit(
        fn=chat_instance.chat,
        inputs=[msg_input, reasoning_mode, temperature, max_tokens, chatbot],
        outputs=[chatbot],
    ).then(lambda: "", None, msg_input)

    reset_btn.click(
        fn=chat_instance.reset,
        inputs=[],
        outputs=[chatbot],
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
    )
