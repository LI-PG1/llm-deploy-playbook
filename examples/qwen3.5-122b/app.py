# ============================================
# Qwen3.5-122B-A10B Gradio Chat WebUI
#
# 支持两种推理引擎：
#   1. vLLM（默认，更快）：设置 VLLM_INFERENCE=1
#   2. Transformers（回退）：默认路径
#
# 关键设计原则：
#   模块级不 import torch/transformers/vllm！
#   所有重型导入放在 load_model() 内部，避免
#   与平台 GPU RPC 初始化模式冲突。
# ============================================

import os
import json
import time
import ssl
import urllib.request
from threading import Lock, Event, Thread

MODEL_PATH = os.environ.get("MODEL_PATH", "/gemini/pretrain/Qwen3.5-122B-A10B")
INFERENCE_MODE = os.environ.get("VLLM_INFERENCE", "")

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_SSL_VERIFY"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

MODEL = None
TOKENIZER = None
MODEL_LOCK = Lock()
MODEL_LOADED = Event()

# ==================== 保活线程 ====================

def _keepalive_thread():
    """模型加载期间每 5 分钟访问 Gradio HTTP 端口，防止平台空闲超时杀进程。"""
    while not MODEL_LOADED.is_set():
        print(f"[KEEPALIVE] Model loading in progress... ({time.strftime('%H:%M:%S')})", flush=True)
        try:
            urllib.request.urlopen("http://127.0.0.1:7860", timeout=10)
        except Exception:
            pass
        MODEL_LOADED.wait(300)

# ==================== 模型加载 ====================

def init_vllm():
    """初始化 vLLM 引擎."""
    from vllm import LLM

    print(f"[VLLM] Initializing vLLM with model from {MODEL_PATH}...")
    t0 = time.time()
    llm = LLM(
        model=MODEL_PATH,
        trust_remote_code=True,
        tensor_parallel_size=1,
        dtype="bfloat16",
    )
    elapsed = time.time() - t0
    print(f"[VLLM] Model loaded in {elapsed:.1f}s")
    return llm

def load_model():
    global MODEL, TOKENIZER
    if MODEL is not None:
        return MODEL, TOKENIZER

    with MODEL_LOCK:
        if MODEL is not None:
            return MODEL, TOKENIZER

        # 启动保活线程
        keepalive = Thread(target=_keepalive_thread, daemon=True)
        keepalive.start()

        import torch

        if INFERENCE_MODE == "1":
            # ========== vLLM 路径 ==========
            from transformers import AutoTokenizer

            print(f"[LOAD] Loading tokenizer from {MODEL_PATH}...")
            TOKENIZER = AutoTokenizer.from_pretrained(
                MODEL_PATH,
                trust_remote_code=True,
                local_files_only=True,
            )
            MODEL = init_vllm()
        else:
            # ========== Transformers 路径（回退）==========
            from transformers import AutoModelForCausalLM, AutoTokenizer

            print(f"[LOAD] Loading tokenizer from {MODEL_PATH}...")
            TOKENIZER = AutoTokenizer.from_pretrained(
                MODEL_PATH,
                trust_remote_code=True,
                local_files_only=True,
            )

            print(f"[LOAD] Loading model from {MODEL_PATH}...")
            print(f"[LOAD] 122B total / 10B active MoE model")
            t0 = time.time()
            MODEL = AutoModelForCausalLM.from_pretrained(
                MODEL_PATH,
                dtype=torch.bfloat16,
                trust_remote_code=True,
                local_files_only=True,
                device_map="auto",
            )
            elapsed = time.time() - t0
            print(f"[LOAD] Model loaded in {elapsed:.1f}s")
            print(f"[LOAD] Device: cuda")

        MODEL_LOADED.set()
        return MODEL, TOKENIZER

# ==================== vLLM 推理 ====================

def _generate_vllm(llm, tokenizer, messages, max_new_tokens, temperature, top_p):
    """使用 vLLM 引擎生成."""
    from vllm import SamplingParams

    sampling_params = SamplingParams(
        temperature=temperature if temperature > 0 else 0.0,
        top_p=top_p if temperature > 0 else 1.0,
        max_tokens=max_new_tokens,
    )

    t0 = time.time()
    outputs = llm.chat(messages, sampling_params=sampling_params, use_tqdm=False)
    elapsed = time.time() - t0

    response = outputs[0].outputs[0].text
    output_tokens = len(outputs[0].outputs[0].token_ids)

    print(f"[VLLM] Output tokens: {output_tokens}, time: {elapsed:.1f}s, speed: {output_tokens/elapsed:.1f} tok/s")
    return response

# ==================== Transformers 推理 ====================

def _generate_transformers(model, tokenizer, messages, max_new_tokens, temperature, top_p):
    """使用 Transformers 生成（回退方案）."""
    import torch

    tokenized = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
    )

    tokenized = tokenized.to(model.device)
    inputs = tokenized.input_ids
    attention_mask = tokenized.attention_mask
    input_len = inputs.shape[-1]
    print(f"[TRANSFORMERS] Input tokens: {input_len}")

    t0 = time.time()
    with torch.no_grad():
        generated = model.generate(
            inputs,
            attention_mask=attention_mask,
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

    print(f"[TRANSFORMERS] Output tokens: {len(output_tokens)}, time: {elapsed:.1f}s, speed: {len(output_tokens)/elapsed:.1f} tok/s")
    return response

# ==================== 统一生成入口 ====================

def generate_response(messages, max_new_tokens=2048, temperature=0.7, top_p=0.8):
    model, tokenizer = load_model()

    if INFERENCE_MODE == "1":
        return _generate_vllm(model, tokenizer, messages, max_new_tokens, temperature, top_p)
    else:
        return _generate_transformers(model, tokenizer, messages, max_new_tokens, temperature, top_p)

# ==================== 聊天类 ====================

class Qwen35Chat:
    def __init__(self):
        self.system_prompt = ""

    def reset(self):
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

        # history is already list of dicts with role/content keys
        messages.extend(history)
        messages.append({"role": "user", "content": user_message})

        try:
            response = generate_response(
                messages=messages,
                max_new_tokens=int(max_new_tokens),
                temperature=float(temperature),
                top_p=float(top_p),
            )

            history.append({"role": "user", "content": user_message})
            if response.strip():
                history.append({"role": "assistant", "content": response.strip()})
            else:
                history.append({"role": "assistant", "content": "(empty response)"})

        except Exception as e:
            error_msg = f"Error: {str(e)}"
            print(error_msg)
            history.append({"role": "user", "content": user_message})
            history.append({"role": "assistant", "content": error_msg})

        return history, history

# ==================== 构建 Gradio UI ====================

def create_demo():
    import gradio as gr

    chat_instance = Qwen35Chat()

    engine_label = "vLLM" if INFERENCE_MODE == "1" else "Transformers"
    with gr.Blocks(title=f"Qwen3.5-122B-A10B Chat ({engine_label})") as demo:
        gr.Markdown(f"""
        # Qwen3.5-122B-A10B Chat

        Alibaba Qwen3.5 MoE model: **122B** total / 10B active parameters.
        Inference engine: **{engine_label}**
        Gated DeltaNet + Mixture-of-Experts architecture. Apache 2.0 license.

        Model: [Qwen/Qwen3.5-122B-A10B](https://huggingface.co/Qwen/Qwen3.5-122B-A10B)
        """)

        with gr.Row():
            with gr.Column(scale=1):
                with gr.Accordion("System Prompt", open=False):
                    system_input = gr.Textbox(
                        label="System Prompt",
                        placeholder="You are Qwen, a helpful AI assistant...",
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

    return demo

# ==================== 启动入口 ====================

if __name__ == "__main__":
    print(f"[START] Inference engine: {'vLLM' if INFERENCE_MODE == '1' else 'Transformers'}")
    print(f"[START] Model path: {MODEL_PATH}")

    # 先启动后台模型加载线程
    def _load_model_thread():
        print("[START] Pre-loading model in background thread...")
        try:
            load_model()
            print("[START] Model ready")
        except Exception as e:
            print(f"[START] Model loading FAILED: {e}")

    Thread(target=_load_model_thread, daemon=True).start()

    # 再启动 Gradio（theme 移入 launch() 适配 Gradio 6.0）
    import gradio as gr
    demo = create_demo()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        theme=gr.themes.Soft(),
    )
