import gradio as gr
import os
import torch
import time
import ssl
import urllib.request
import json
from threading import Lock

MODEL_PATH = os.environ.get("MODEL_PATH", "/gemini/pretrain/Phi-4-mini-instruct-model")

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_SSL_VERIFY"] = "1"

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
            torch_dtype=torch.float16,
            local_files_only=True,
        )
        MODEL = MODEL.cuda()
        elapsed = time.time() - t0
        print(f"[LOAD] Model loaded in {elapsed:.1f}s")
        print(f"[LOAD] Device: cuda")

        return MODEL, TOKENIZER


def generate_response(messages, max_new_tokens=2048, temperature=0.7, top_p=0.8):
    model, tokenizer = load_model()

    try:
        tokenized = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
        )
    except TypeError:
        tokenized = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
        )

    tokenized = tokenized.to(model.device)
    input_ids = tokenized["input_ids"]
    input_len = input_ids.shape[-1]
    print(f"[INFER] Input tokens: {input_len}")

    t0 = time.time()
    with torch.no_grad():
        generated = model.generate(
            **tokenized,
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


class Phi4Chat:
    def __init__(self):
        self.conversation = []
        self.system_prompt = ""

    def reset(self):
        self.conversation = []
        return [], []

    def set_system_prompt(self, system_text):
        self.system_prompt = system_text.strip()
        return f"System prompt set ({len(self.system_prompt)} chars)"

    def chat(self, user_message, history, temperature, top_p, max_new_tokens):
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
            )

            if response.strip():
                history.append((user_message, response.strip()))
            else:
                history.append((user_message, "(empty response)"))

        except Exception as e:
            import traceback
            error_msg = f"Error: {type(e).__name__}: {e}"
            print(error_msg)
            traceback.print_exc()
            history.append((user_message, error_msg))

        return history, history


chat_instance = Phi4Chat()


with gr.Blocks(title="Phi-4-mini-instruct Chat", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # Phi-4-mini-instruct Chat

    Microsoft Phi-4-mini-instruct: compact instruction-tuned language model.
    128K context window. MIT license.

    Model: [microsoft/Phi-4-mini-instruct](https://huggingface.co/microsoft/Phi-4-mini-instruct)
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
        inputs=[msg_input, state, temperature_slider, top_p_slider, max_tokens_slider],
        outputs=[chatbot, state],
    ).then(lambda: "", None, msg_input)

    msg_input.submit(
        fn=chat_instance.chat,
        inputs=[msg_input, state, temperature_slider, top_p_slider, max_tokens_slider],
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
