"""
DeepSeek-R1-Distill-Qwen-14B 推理服务示例
=========================================
特点：中大型模型、多卡 device_map、CoT 推理展示
显存：~27GB (BF16)，建议 2 卡以上
对应文档：docs/07-MoE模型 / docs/08-代码适配 / docs/10-常见错误排查

踩坑提示（自查清单）：
  - huggingface_hub validate_repo_id 不认本地路径 → 需 monkey-patch
  - eos_token_id 为列表时需原样传递
  - 模型目录名含连字符需 transformers >= 5.7

启动方式：
    export MODEL_PATH=/path/to/model
    python app.py
"""

import os
import sys
import re
import time
import torch
import gradio as gr
from threading import Lock

MODEL_PATH = os.environ.get("MODEL_PATH", "/models/DeepSeek-R1-Distill-Qwen-14B")

# ===================== monkey-patches =====================

# huggingface_hub 1.x 不认绝对路径
from huggingface_hub.utils._validators import validate_repo_id
import huggingface_hub.utils._validators as _v
_v.validate_repo_id = lambda x, **kwargs: None

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# ===================== 全局单例加载 =====================

model = None
tokenizer = None
model_lock = Lock()

def load_model():
    global model, tokenizer

    if model is not None:
        return model, tokenizer

    with model_lock:
        if model is not None:
            return model, tokenizer

        from transformers import AutoModelForCausalLM, AutoTokenizer

        t0 = time.time()

        print(f"[1/2] 加载 tokenizer: {MODEL_PATH}")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)

        print(f"[2/2] 加载 model: {MODEL_PATH}  torch.bfloat16")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.bfloat16,  # 14B 模型 BF16 约 27GB
            device_map="auto",           # 自动分布到多 GPU
            local_files_only=True,
        )

        elapsed = time.time() - t0
        gpu_mem = torch.cuda.memory_allocated() / 1024**3
        print(f"[完成] 耗时 {elapsed:.1f}s，显存 {gpu_mem:.1f}GB，device: {model.device}")
        return model, tokenizer

# ===================== CoT 推理 / 回答分离 =====================

def split_thinking(text):
    """从 DeepSeek-R1 的 CoT 输出中分离思考过程和最终回答。"""
    patterns = [
        r"^(.*?)<｜end▁of▁thinking｜>\s*(.*)",       # 英文版
        r"^(.*?)回答\s*(.*)",            # 中文版
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match and match.group(1).strip() and match.group(2).strip():
            return match.group(1).strip(), match.group(2).strip()
    return "", text

# ===================== 推理 =====================

def generate_response(messages, max_new_tokens=4096, temperature=0.6, top_p=0.95):
    model, tokenizer = load_model()

    tokenized = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
    )
    tokenized = tokenized.to(model.device)
    input_len = tokenized["input_ids"].shape[-1]

    print(f"[推理] 输入 {input_len} tokens")

    t0 = time.time()
    with torch.no_grad():
        generated = model.generate(
            **tokenized,
            max_new_tokens=int(max_new_tokens),
            temperature=temperature if temperature > 0 else None,
            top_p=top_p if temperature > 0 else None,
            do_sample=temperature > 0,
            repetition_penalty=1.05,     # 减少重复
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,  # 列表原样传递
        )
    elapsed = time.time() - t0
    output_tokens = generated[0][input_len:]
    tok_speed = len(output_tokens) / elapsed if elapsed > 0 else 0
    print(f"[推理] 输出 {len(output_tokens)} tokens，耗时 {elapsed:.1f}s，速度 {tok_speed:.1f} tok/s")

    return tokenizer.decode(output_tokens, skip_special_tokens=True)

# ===================== Gradio 界面 =====================

class ChatBot:
    def __init__(self):
        self.conversation = []
        self.system_prompt = ""

    def reset(self):
        self.conversation = []
        return []

    def set_system_prompt(self, text):
        self.system_prompt = text.strip()
        return f"System prompt 已设置（{len(self.system_prompt)} 字符）"

    def chat(self, user_message, temperature, top_p, max_new_tokens, show_thinking):
        if not user_message.strip():
            return self.conversation

        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        for q, a in self.conversation:
            messages.append({"role": "user", "content": q})
            messages.append({"role": "assistant", "content": a})
        messages.append({"role": "user", "content": user_message})

        try:
            raw = generate_response(messages, max_new_tokens, temperature, top_p)
            thinking, answer = split_thinking(raw)

            if show_thinking and thinking:
                display = f"**思考过程**\n{thinking}\n\n**回答**\n{answer}"
            else:
                display = answer or raw

            self.conversation.append((user_message, display.strip() or "(empty)"))

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.conversation.append((user_message, f"Error: {type(e).__name__}: {e}"))

        return self.conversation

bot = ChatBot()

with gr.Blocks(title="DeepSeek-R1-14B Chat", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""# DeepSeek-R1-Distill-Qwen-14B Chat\n\nDeepSeek 推理模型，蒸馏到 Qwen2-14B（BF16 ~27GB），支持 Thinking 显示""")

    with gr.Row():
        with gr.Column(scale=1):
            with gr.Accordion("System Prompt", open=False):
                system_input = gr.Textbox(label="System Prompt", lines=3,
                    placeholder="You are a helpful AI assistant...")
                set_system_btn = gr.Button("设置", size="sm")

            with gr.Accordion("生成参数", open=False):
                temperature = gr.Slider(0.0, 2.0, 0.6, step=0.1, label="Temperature")
                top_p = gr.Slider(0.1, 1.0, 0.95, step=0.05, label="Top-P")
                max_tokens = gr.Slider(512, 16384, 4096, step=256, label="Max New Tokens")
                show_thinking = gr.Checkbox(True, label="显示思考过程")

            reset_btn = gr.Button("清空对话", variant="secondary")

        with gr.Column(scale=3):
            chatbot = gr.Chatbot(height=500, bubble_full_width=False)
            msg = gr.Textbox(label="输入消息", placeholder="输入消息后回车...", lines=2)
            send_btn = gr.Button("发送", variant="primary")

    send_btn.click(bot.chat, [msg, temperature, top_p, max_tokens, show_thinking], [chatbot])
    msg.submit(bot.chat, [msg, temperature, top_p, max_tokens, show_thinking], [chatbot])
    reset_btn.click(bot.reset, [], [chatbot])

    set_system_btn.click(bot.set_system_prompt, [system_input], [])
    set_system_btn.click(lambda m: gr.Info(message=m, duration=3), [system_input], [])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
