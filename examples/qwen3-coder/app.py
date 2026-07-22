"""
Qwen3-Coder-30B-A3B-Instruct 代码生成聊天界面
=============================================
30B MoE (3.3B active) | 256K context | Code Generation
Recommended: temperature=0.7, top_p=0.8, top_k=20, repetition_penalty=1.05
"""
import gradio as gr
import os
import torch
import time
from threading import Lock

MODEL_PATH = os.environ.get("MODEL_PATH", "/gemini/pretrain/Qwen3-Coder-30B-A3B-Instruct-model")

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_SSL_VERIFY"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# 确保 LD_LIBRARY_PATH 包含 orion CUDA 库
import os as _os
_os.environ.setdefault("LD_LIBRARY_PATH", "")
if "/opt/orion" not in _os.environ.get("LD_LIBRARY_PATH", ""):
    _os.environ["LD_LIBRARY_PATH"] = "/opt/orion/orion_runtime/gpu/cuda/orion-cuda-12.2/:" + _os.environ.get("LD_LIBRARY_PATH", "")

# Monkey-patch: 绕过 huggingface_hub 对本地绝对路径的校验
import huggingface_hub as _hf
_hf.validate_repo_id = lambda repo_id, **kwargs: None

MODEL = None
TOKENIZER = None
MODEL_LOCK = Lock()

# ===== 推荐参数（官方）=====
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.8
DEFAULT_TOP_K = 20
DEFAULT_REPETITION_PENALTY = 1.05
DEFAULT_MAX_NEW_TOKENS = 65536


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
            dtype=torch.bfloat16,
            trust_remote_code=True,
            local_files_only=True,
            device_map="auto",
        )
        elapsed = time.time() - t0
        print(f"[LOAD] Model loaded in {elapsed:.1f}s")
        print(f"[LOAD] Device: {MODEL.device}")

        return MODEL, TOKENIZER


def generate_response(messages, max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
                      temperature=DEFAULT_TEMPERATURE, top_p=DEFAULT_TOP_P,
                      top_k=DEFAULT_TOP_K, repetition_penalty=DEFAULT_REPETITION_PENALTY):
    """
    Generate code completion response.
    Qwen3-Coder does NOT support thinking mode (no <think> blocks).
    """
    model, tokenizer = load_model()

    # Apply chat template
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

    # 从 config 获取 eos_token_id（可能为列表）
    eos_ids = model.config.eos_token_id
    if isinstance(eos_ids, (list, tuple)):
        eos_ids = eos_ids[0]

    t0 = time.time()
    with torch.no_grad():
        generated = model.generate(
            inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature if temperature > 0 else None,
            top_p=top_p if temperature > 0 else None,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=eos_ids,
        )
    elapsed = time.time() - t0

    output_tokens = generated[0][input_len:]
    response = tokenizer.decode(output_tokens, skip_special_tokens=True)

    tok_speed = len(output_tokens) / elapsed if elapsed > 0 else 0
    print(f"[INFER] Output tokens: {len(output_tokens)}, "
          f"time: {elapsed:.1f}s, speed: {tok_speed:.1f} tok/s")

    return response, len(output_tokens), elapsed


# ===== Gradio Chat Interface =====
def chat_fn(message, history):
    """Handle chat message - streaming mode for code generation."""
    model, tokenizer = load_model()

    # Build messages from history
    messages = []
    for user_msg, asst_msg in history:
        messages.append({"role": "user", "content": user_msg})
        if asst_msg is not None:
            messages.append({"role": "assistant", "content": asst_msg})
    messages.append({"role": "user", "content": message})

    # Apply chat template
    tokenized = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
    )
    tokenized = tokenized.to(model.device)
    inputs = tokenized.input_ids
    input_len = inputs.shape[-1]

    eos_ids = model.config.eos_token_id
    if isinstance(eos_ids, (list, tuple)):
        eos_ids = eos_ids[0]

    # Generate with streaming
    from transformers import TextStreamer

    class GradioStreamer(TextStreamer):
        def __init__(self, tokenizer):
            super().__init__(tokenizer, skip_prompt=True)
            self.token_cache = []
            self.text_so_far = ""

        def on_finalized_text(self, text: str, stream_end: bool = False):
            self.text_so_far += text
            self.token_cache.append(text)

    streamer = GradioStreamer(tokenizer)

    t0 = time.time()
    with torch.no_grad():
        generated = model.generate(
            inputs,
            max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
            temperature=DEFAULT_TEMPERATURE,
            top_p=DEFAULT_TOP_P,
            top_k=DEFAULT_TOP_K,
            repetition_penalty=DEFAULT_REPETITION_PENALTY,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=eos_ids,
            streamer=streamer,
        )
    elapsed = time.time() - t0

    output_tokens = generated[0][input_len:]
    response = tokenizer.decode(output_tokens, skip_special_tokens=True)

    tok_speed = len(output_tokens) / elapsed if elapsed > 0 else 0
    print(f"[INFER] Output tokens: {len(output_tokens)}, "
          f"time: {elapsed:.1f}s, speed: {tok_speed:.1f} tok/s")

    return response


# ===== Build UI =====
CSS = """
#chatbot { height: 60vh; font-size: 14px; }
#title { text-align: center; }
#params { border: 1px solid #ddd; border-radius: 8px; padding: 12px; }
code { background: #f0f0f0; padding: 2px 4px; border-radius: 3px; font-size: 13px; }
pre { background: #1e1e1e; color: #d4d4d4; padding: 12px; border-radius: 6px; overflow-x: auto; }
"""

with gr.Blocks(title="Qwen3-Coder-30B-A3B-Instruct") as demo:
    gr.Markdown(
        "# Qwen3-Coder-30B-A3B-Instruct 代码生成\n"
        "基于 Qwen3 MoE 架构的代码生成模型，30B 参数（3.3B 激活），支持 256K 上下文。",
        elem_id="title",
    )

    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                label="对话",
                elem_id="chatbot",
                placeholder="输入代码问题，例如：写一个 Python 快速排序算法...",
                height=500,
            )
            msg = gr.Textbox(
                label="输入",
                placeholder="输入你的代码问题...",
                lines=2,
            )
            with gr.Row():
                send_btn = gr.Button("发送", variant="primary", scale=2)
                clear_btn = gr.Button("清空对话", scale=1)

        with gr.Column(scale=1):
            with gr.Group(elem_id="params"):
                gr.Markdown("### 生成参数（官方推荐）")
                max_tokens_slider = gr.Slider(
                    minimum=512, maximum=65536, value=65536,
                    step=512, label="最大输出 Token",
                )
                temperature_slider = gr.Slider(
                    minimum=0.1, maximum=2.0, value=DEFAULT_TEMPERATURE,
                    step=0.05, label="Temperature",
                )
                top_p_slider = gr.Slider(
                    minimum=0.1, maximum=1.0, value=DEFAULT_TOP_P,
                    step=0.05, label="Top-P",
                )
                top_k_slider = gr.Slider(
                    minimum=1, maximum=100, value=DEFAULT_TOP_K,
                    step=1, label="Top-K",
                )
                rep_penalty_slider = gr.Slider(
                    minimum=1.0, maximum=2.0, value=DEFAULT_REPETITION_PENALTY,
                    step=0.05, label="Repetition Penalty",
                )
                gr.Markdown(
                    "**提示**：代码生成推荐 temperature=0.7, "
                    "top_p=0.8, top_k=20, repetition_penalty=1.05"
                )

    # 聊天历史
    chat_history = gr.State([])

    def respond(message, history, max_tokens, temperature, top_p, top_k, rep_penalty):
        if not message.strip():
            return "", history

        history.append({"role": "user", "content": message})

        # Generate (非流式，简单可靠)
        try:
            response, out_tokens, elapsed = generate_response(
                [{"role": "user", "content": message}],
                max_new_tokens=int(max_tokens),
                temperature=temperature,
                top_p=top_p,
                top_k=int(top_k),
                repetition_penalty=rep_penalty,
            )
            history.append({"role": "assistant", "content": response})
        except Exception as e:
            err_msg = f"**错误**: {str(e)}"
            history.append({"role": "assistant", "content": err_msg})
            print(f"[ERROR] {e}")

        return "", history

    def clear_chat():
        return [], []

    # 绑定事件
    send_event = msg.submit(
        respond,
        [msg, chat_history, max_tokens_slider, temperature_slider,
         top_p_slider, top_k_slider, rep_penalty_slider],
        [msg, chatbot],
    )
    send_btn.click(
        respond,
        [msg, chat_history, max_tokens_slider, temperature_slider,
         top_p_slider, top_k_slider, rep_penalty_slider],
        [msg, chatbot],
    )
    clear_btn.click(clear_chat, None, [chatbot, chat_history])


if __name__ == "__main__":
    # 后台预加载模型
    import threading
    threading.Thread(target=load_model, daemon=True).start()

    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
        css=CSS,
        theme=gr.themes.Soft(),
    )
