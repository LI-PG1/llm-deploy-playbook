import os as _os
_os.environ.setdefault("LD_LIBRARY_PATH", "")
if "/opt/orion" not in _os.environ.get("LD_LIBRARY_PATH", ""):
    _os.environ["LD_LIBRARY_PATH"] = "/opt/orion/orion_runtime/gpu/cuda/orion-cuda-12.2/:" + _os.environ.get("LD_LIBRARY_PATH", "")

_os.environ["TRANSFORMERS_OFFLINE"] = "1"

from huggingface_hub.utils._validators import validate_repo_id
import huggingface_hub.utils._validators as _v
_v.validate_repo_id = lambda x, **kwargs: None

import gradio as gr
import re
import torch
import time
from threading import Lock

MODEL_PATH = _os.environ.get("MODEL_PATH", "/gemini/pretrain")

_os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
_os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
_os.environ["HF_HUB_DISABLE_SSL_VERIFY"] = "1"

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
            local_files_only=True,
        )

        print(f"[LOAD] Loading model from {MODEL_PATH}...")
        t0 = time.time()
        MODEL = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            dtype=torch.bfloat16,
            device_map="auto",
            local_files_only=True,
        )
        elapsed = time.time() - t0
        print(f"[LOAD] Model loaded in {elapsed:.1f}s")
        print(f"[LOAD] Device: {MODEL.device}")

        return MODEL, TOKENIZER


def split_thinking(text):
    think_patterns = [
        r"^(.*?) response\s*(.*)",
        r"^(.*?)<｜end▁of▁thinking｜>\s*(.*)",
    ]

    for pattern in think_patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            thinking = match.group(1).strip()
            answer = match.group(2).strip()
            if thinking and answer:
                return thinking, answer

    return "", text


def generate_response(messages, max_new_tokens=4096, temperature=0.6, top_p=0.95):
    model, tokenizer = load_model()

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
            repetition_penalty=1.05,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    elapsed = time.time() - t0

    output_tokens = generated[0][input_len:]
    response = tokenizer.decode(output_tokens, skip_special_tokens=True)

    tokens_per_sec = len(output_tokens) / elapsed if elapsed > 0 else 0
    print(f"[INFER] Output tokens: {len(output_tokens)}, time: {elapsed:.1f}s, speed: {tokens_per_sec:.1f} tok/s")

    return response


class DeepSeekChat:
    def __init__(self):
        self.conversation = []
        self.system_prompt = ""

    def reset(self):
        self.conversation = []
        return []

    def set_system_prompt(self, system_text):
        self.system_prompt = system_text.strip()
        return f"System prompt set ({len(self.system_prompt)} chars)"

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
            raw_response = generate_response(
                messages=messages,
                max_new_tokens=int(max_new_tokens),
                temperature=float(temperature),
                top_p=float(top_p),
            )

            thinking, answer = split_thinking(raw_response)

            display_text = raw_response
            if show_thinking and thinking:
                display_text = f"**[Thinking Process]**\n```\n{thinking}\n```\n\n**[Response]**\n{answer}"
            elif not show_thinking and answer:
                display_text = answer

            if display_text.strip():
                self.conversation.append((user_message, display_text.strip()))
            else:
                self.conversation.append((user_message, "(empty response)"))

        except Exception as e:
            import traceback
            error_msg = f"Error: {type(e).__name__}: {e}"
            print(error_msg)
            traceback.print_exc()
            self.conversation.append((user_message, error_msg))

        return self.conversation


chat_instance = DeepSeekChat()


with gr.Blocks(title="DeepSeek-R1-Distill-Qwen-14B Chat", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # DeepSeek-R1-Distill-Qwen-14B Chat

    **DeepSeek-R1** reasoning model distilled to Qwen2-14B (BF16, ~27GB total). This model generates chain-of-thought reasoning before producing the final answer. Supports Think mode toggle.

    Model: [deepseek-ai/DeepSeek-R1-Distill-Qwen-14B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B)
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

            with gr.Accordion("Generation Parameters", open=False):
                temperature_slider = gr.Slider(
                    0.0, 2.0, 0.6, step=0.1,
                    label="Temperature (0 = deterministic)",
                )
                top_p_slider = gr.Slider(
                    0.1, 1.0, 0.95, step=0.05,
                    label="Top-P",
                )
                max_tokens_slider = gr.Slider(
                    512, 16384, 4096, step=256,
                    label="Max New Tokens",
                )
                show_thinking_cb = gr.Checkbox(
                    True,
                    label="Show Thinking Process",
                    info="Display model's chain-of-thought reasoning",
                )

            reset_btn = gr.Button("Clear Conversation", variant="secondary")

        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                label="Conversation",
                height=500,
                bubble_full_width=False,
            )
            msg_input = gr.Textbox(
                label="Your Message",
                placeholder="Type your message and press Enter...",
                lines=2,
            )
            send_btn = gr.Button("Send", variant="primary", size="lg")

    send_btn.click(
        fn=chat_instance.chat,
        inputs=[msg_input, temperature_slider, top_p_slider, max_tokens_slider, show_thinking_cb],
        outputs=[chatbot],
    ).then(lambda: "", None, msg_input)

    msg_input.submit(
        fn=chat_instance.chat,
        inputs=[msg_input, temperature_slider, top_p_slider, max_tokens_slider, show_thinking_cb],
        outputs=[chatbot],
    ).then(lambda: "", None, msg_input)

    reset_btn.click(
        fn=chat_instance.reset,
        inputs=[],
        outputs=[chatbot],
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
