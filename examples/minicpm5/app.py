import gradio as gr
import os
import torch
import time
from threading import Lock

MODEL_PATH = os.environ.get("MODEL_PATH", "/gemini/pretrain/MiniCPM5-1B-model")

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_SSL_VERIFY"] = "1"

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
            local_files_only=False,
        )

        print(f"[LOAD] Loading model from {MODEL_PATH}...")
        t0 = time.time()
        MODEL = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.bfloat16,
            local_files_only=False,
        )
        MODEL = MODEL.cuda()
        elapsed = time.time() - t0
        print(f"[LOAD] Model loaded in {elapsed:.1f}s")
        print(f"[LOAD] Device: cuda")

        return MODEL, TOKENIZER


def generate_response(messages, max_new_tokens=512, temperature=0.7, top_p=0.95, enable_thinking=False):
    model, tokenizer = load_model()

    tokenized = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        enable_thinking=enable_thinking,
    )
    tokenized = {k: v.to(model.device) for k, v in tokenized.items()}
    input_len = tokenized["input_ids"].shape[-1]
    print(f"[INFER] Input tokens: {input_len}, thinking={enable_thinking}")

    t0 = time.time()
    with torch.no_grad():
        generated = model.generate(
            **tokenized,
            max_new_tokens=max_new_tokens,
            temperature=temperature if temperature > 0 else None,
            top_p=top_p if temperature > 0 else None,
            do_sample=temperature > 0.01,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    elapsed = time.time() - t0

    output_tokens = generated[0][input_len:]
    response = tokenizer.decode(output_tokens, skip_special_tokens=True)

    tok_s = len(output_tokens) / elapsed if elapsed > 0 else 0
    print(f"[INFER] Output tokens: {len(output_tokens)}, time: {elapsed:.1f}s, speed: {tok_s:.1f} tok/s")
    return response


class MiniCPM5Chat:
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


chat_instance = MiniCPM5Chat()


with gr.Blocks(title="MiniCPM5-1B Chat", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # MiniCPM5-1B Intelligent Chat

    **OpenBMB MiniCPM5-1B**: 1B-class open-source SOTA dense Transformer.
    Built for on-device deployment with Think / No-Think hybrid reasoning modes.
    131K context length. Standard LlamaForCausalLM architecture — no custom fork needed.

    Model: [openbmb/MiniCPM5-1B](https://huggingface.co/openbmb/MiniCPM5-1B)
    """)

    with gr.Row():
        with gr.Column(scale=1):
            with gr.Accordion("System Prompt", open=False):
                system_input = gr.Textbox(
                    label="System Prompt",
                    placeholder="You are MiniCPM5, a helpful AI assistant...",
                    lines=3,
                )
                set_system_btn = gr.Button("Set System Prompt", size="sm")

            thinking_toggle = gr.Checkbox(
                label="Enable Thinking Mode (step-by-step reasoning with <think> tags)",
                value=False,
            )

            with gr.Accordion("Generation Parameters", open=False):
                temperature_slider = gr.Slider(
                    0.0, 2.0, 0.7, step=0.05,
                    label="Temperature (No Think: 0.7 / Think: 0.9 recommended)",
                )
                top_p_slider = gr.Slider(
                    0.1, 1.0, 0.95, step=0.05,
                    label="Top-P",
                )
                max_tokens_slider = gr.Slider(
                    64, 4096, 512, step=64,
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
