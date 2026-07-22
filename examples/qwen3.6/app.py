import gradio as gr
import os
import torch
import time
import ssl
import urllib.request
import json
from threading import Lock

MODEL_PATH = os.environ.get("MODEL_PATH", "/gemini/pretrain/Qwen3.6-35B-A3B")

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_SSL_VERIFY"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import os as _os
_os.environ.setdefault("LD_LIBRARY_PATH", "")
if "/opt/orion" not in _os.environ.get("LD_LIBRARY_PATH", ""):
    _os.environ["LD_LIBRARY_PATH"] = "/opt/orion/orion_runtime/gpu/cuda/orion-cuda-12.2/:" + _os.environ.get("LD_LIBRARY_PATH", "")

MODEL = None
TOKENIZER = None
MODEL_LOCK = Lock()

def load_model():
    global MODEL, TOKENIZER
    if MODEL is not None:
        return MODEL, TOKENIZER

    with MODEL_LOCK:
        if MODEL is not None:
            return MODEL, TOKENIZER

        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"[LOAD] Loading tokenizer from {MODEL_PATH}...")
        TOKENIZER = AutoTokenizer.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True,
            local_files_only=True,
        )

        print(f"[LOAD] Loading model from {MODEL_PATH}...")
        t0 = time.time()
        MODEL = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            local_files_only=True,
            device_map="auto",
        )
        elapsed = time.time() - t0
        print(f"[LOAD] Model loaded in {elapsed:.1f}s")
        print(f"[LOAD] Device: cuda")

        return MODEL, TOKENIZER

def generate_response(messages, max_new_tokens=2048, temperature=0.7, top_p=0.8, enable_thinking=True):
    model, tokenizer = load_model()

    try:
        tokenized = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            enable_thinking=enable_thinking,
        )
    except TypeError:
        tokenized = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
        )

    tokenized = tokenized.to(model.device)
    inputs = tokenized.input_ids
    input_len = inputs.shape[-1]
    print(f"[INFER] Input tokens: {input_len}")

    t0 = time.time()
    with torch.no_grad():
        generated = model.generate(
            inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature if temperature > 0 else None,
            top_p=top_p if temperature > 0 else None,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    elapsed = time.time() - t0

    output_tokens = generated[0][input_len:]
    response = tokenizer.decode(output_tokens, skip_special_tokens=True)

    print(f"[INFER] Output tokens: {len(output_tokens)}, time: {elapsed:.1f}s, speed: {len(output_tokens)/elapsed:.1f} tok/s")
    return response

class Qwen3Chat:
    def __init__(self):
        self.conversation = []
        self.system_prompt = ""

    def reset(self):
        self.conversation = []
        return [], []

    def set_system_prompt(self, system_text):
        self.system_prompt = system_text.strip()
        return f"System prompt set ({len(self.system_prompt)} chars)"

    def chat(self, user_message, history, temperature, top_p, max_new_tokens, enable_thinking):
        if not user_message.strip():
            return history, history

        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})

        for h in history:
            if h[0]:
                messages.append({"role": "user", "content": h[0]})
            if h[1]:
                messages.append({"role": "assistant", "content": h[1]})

        messages.append({"role": "user", "content": user_message})

        try:
            response = generate_response(
                messages=messages,
                max_new_tokens=int(max_new_tokens),
                temperature=float(temperature),
                top_p=float(top_p),
                enable_thinking=enable_thinking,
            )

            if response.strip():
                history.append((user_message, response.strip()))
            else:
                history.append((user_message, "(empty response)"))

        except Exception as e:
            error_msg = f"Error: {str(e)}"
            print(error_msg)
            history.append((user_message, error_msg))

        return history, history

chat_instance = Qwen3Chat()

with gr.Blocks(title="Qwen3.6-35B-A3B Chat", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # Qwen3.6-35B-A3B Intelligent Chat

    Alibaba Qwen3.6 MoE model: 35B total / 3B active parameters.
    Supports text chat with thinking mode. Apache 2.0 license.

    Model: [Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)
    """)

    with gr.Row():
        with gr.Column(scale=1):
            with gr.Accordion("System Prompt", open=False):
                system_input = gr.Textbox(
                    label="System Prompt",
                    placeholder="You are a helpful AI assistant...",
                    lines=3,
                )
                set_system_btn = gr.Button("Set System Prompt", size="sm")

            thinking_toggle = gr.Checkbox(
                label="Enable Thinking Mode (step-by-step reasoning)",
                value=True,
            )

            with gr.Accordion("Generation Parameters", open=False):
                temperature_slider = gr.Slider(
                    0.0, 2.0, 0.7, step=0.1,
                    label="Temperature (0 = deterministic)",
                )
                top_p_slider = gr.Slider(
                    0.1, 1.0, 0.8, step=0.05,
                    label="Top-P",
                )
                max_tokens_slider = gr.Slider(
                    128, 8192, 2048, step=128,
                    label="Max New Tokens",
                )

            reset_btn = gr.Button("Clear Conversation", variant="secondary")

        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                label="Conversation",
                height=550,
                bubble_full_width=False,
            )
            msg_input = gr.Textbox(
                label="Your Message",
                placeholder="Type your message and press Enter...",
                lines=2,
            )
            send_btn = gr.Button("Send", variant="primary", size="lg")

    state = gr.State([])

    send_btn.click(
        fn=chat_instance.chat,
        inputs=[msg_input, state, temperature_slider, top_p_slider, max_tokens_slider, thinking_toggle],
        outputs=[chatbot, state],
    ).then(lambda: "", None, msg_input)

    msg_input.submit(
        fn=chat_instance.chat,
        inputs=[msg_input, state, temperature_slider, top_p_slider, max_tokens_slider, thinking_toggle],
        outputs=[chatbot, state],
    ).then(lambda: "", None, msg_input)

    reset_btn.click(
        fn=chat_instance.reset,
        inputs=[],
        outputs=[chatbot, state],
    )

    sys_msg_status = gr.Textbox(label="", value="", visible=False)

    set_system_btn.click(
        fn=chat_instance.set_system_prompt,
        inputs=[system_input],
        outputs=[sys_msg_status],
    ).then(
        fn=lambda msg: gr.Info(message=msg, duration=3),
        inputs=[sys_msg_status],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
