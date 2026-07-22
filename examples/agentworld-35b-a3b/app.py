"""
Qwen-AgentWorld-35B-A3B - Agent Environment Simulator WebUI
===========================================================
7 domains: MCP, Search, Terminal, SWE, Android, Web, OS
Model: Qwen/Qwen-AgentWorld-35B-A3B (35B MoE, 3B active)
"""
import ctypes as _ct
import os as _os
import subprocess as _sp
import sys as _sys

# 自动安装 torch 缺失的 nvidia-nccl-cu12 子包（镜像 --no-deps 构建时未打包）
_nccl_path = "/root/miniconda3/lib/python3.11/site-packages/nvidia/nccl/lib/libnccl.so.2"
if not _os.path.exists(_nccl_path):
    print("[SETUP] nvidia-nccl-cu12 未安装，正在安装...")
    _sp.run([_sys.executable, "-m", "pip", "install", "nvidia-nccl-cu12",
             "--no-deps"])
# 预加载 NCCL（含 ncclCommResume），先于 gradio/torch 加载
if _os.path.exists(_nccl_path):
    print("[SETUP] 预加载 torch 自带 NCCL...")
    _ct.cdll.LoadLibrary(_nccl_path)

import gradio as gr
import os
import time
from threading import Lock

MODEL_PATH = os.environ.get("MODEL_PATH", "/gemini/pretrain/models")

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_SSL_VERIFY"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Monkey-patch: bypass validate_repo_id rejecting absolute local paths
# (huggingface_hub 1.x refuses /gemini/pretrain/... paths)
import huggingface_hub as _hf
_hf.validate_repo_id = lambda repo_id, **kwargs: None

# ===== Domain System Prompts =====
DOMAIN_PROMPTS = {
    "MCP (Tool Calling)": (
        "You are a language world model simulating an MCP (Model Context Protocol) environment. "
        "Given the agent's tool call and arguments, predict the tool execution result, including return values, errors, and side effects."
    ),
    "Search": (
        "You are a language world model simulating a web search engine environment. "
        "Given the agent's search query, predict the search results page, including snippets, URLs, and pagination."
    ),
    "Terminal": (
        "You are a language world model simulating a Linux terminal environment. "
        "Given the user's shell command, predict the exact terminal output (stdout, stderr, exit code)."
    ),
    "SWE (Software Engineering)": (
        "You are a language world model simulating a software development environment. "
        "Given the agent's code edit, predict the file diff, compilation output, and test results."
    ),
    "Android": (
        "You are a language world model simulating an Android device environment. "
        "Given the agent's UI action (tap, swipe, type), predict the screen state change and app response."
    ),
    "Web": (
        "You are a language world model simulating a web browser environment. "
        "Given the agent's browser action (click, navigate, input), predict the rendered page state and DOM changes."
    ),
    "OS (Desktop)": (
        "You are a language world model simulating a desktop operating system environment. "
        "Given the agent's OS-level action (file operations, application launch, system commands), predict the system state change."
    ),
}

def get_system_prompt(domain_key):
    """Return system prompt for the selected domain."""
    return DOMAIN_PROMPTS.get(domain_key, DOMAIN_PROMPTS["Terminal"])

# ===== Model Loading =====
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

        import torch
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
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            local_files_only=True,
            device_map="auto",
        )
        elapsed = time.time() - t0
        print(f"[LOAD] Model loaded in {elapsed:.1f}s")
        print(f"[LOAD] Device: {MODEL.device}")

        return MODEL, TOKENIZER

def generate_response(messages, max_new_tokens=4096, temperature=0.6, top_p=0.95, top_k=20):
    """
    Generate simulation response.
    Recommended params (from Qwen README):
      temperature=0.6, top_p=0.95, top_k=20, max_new_tokens up to 32768
    """
    import torch
    model, tokenizer = load_model()

    try:
        tokenized = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            enable_thinking=True,
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

    # 从模型 config 获取正确的 eos_token_id（可能与 tokenizer 不同）
    eos_ids = model.config.eos_token_id
    if eos_ids is None:
        eos_ids = tokenizer.eos_token_id
    if not isinstance(eos_ids, list):
        eos_ids = [eos_ids]

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = eos_ids[0]  # 降级保障

    t0 = time.time()
    with torch.no_grad():
        generated = model.generate(
            inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=True,
            pad_token_id=pad_id,
            eos_token_id=eos_ids,
        )
    elapsed = time.time() - t0

    output_tokens = generated[0][input_len:]
    response = tokenizer.decode(output_tokens, skip_special_tokens=True)

    tok_per_sec = len(output_tokens) / elapsed if elapsed > 0 else 0
    print(f"[INFER] Output tokens: {len(output_tokens)}, time: {elapsed:.1f}s, speed: {tok_per_sec:.1f} tok/s")
    return response, len(output_tokens), round(elapsed, 1), round(tok_per_sec, 1)

# ===== Gradio Chat Interface =====
class AgentWorldChat:
    def __init__(self):
        self.conversation = []
        self.current_domain = "Terminal"

    def reset(self):
        self.conversation = []
        return [], []

    def set_domain(self, domain_key):
        self.current_domain = domain_key
        prompt = get_system_prompt(domain_key)
        return prompt

    def chat(self, user_message, history, temperature, top_p, top_k, max_new_tokens):
        if not user_message.strip():
            return history, history, "", "", "", ""

        if history is None:
            history = []

        domain_prompt = get_system_prompt(self.current_domain)
        messages = [{"role": "system", "content": domain_prompt}]

        for h in history:
            messages.append(h)

        messages.append({"role": "user", "content": user_message})

        try:
            response, out_tokens, elapsed, tok_speed = generate_response(
                messages=messages,
                max_new_tokens=int(max_new_tokens),
                temperature=float(temperature),
                top_p=float(top_p),
                top_k=int(top_k),
            )

            if response.strip():
                history.append({"role": "user", "content": user_message})
                history.append({"role": "assistant", "content": response.strip()})
            else:
                history.append({"role": "user", "content": user_message})
                history.append({"role": "assistant", "content": "(empty response)"})

            stats = f"Tokens: {out_tokens} | Time: {elapsed}s | Speed: {tok_speed} tok/s"

        except Exception as e:
            error_msg = f"Error: {str(e)}"
            print(error_msg)
            history.append({"role": "user", "content": user_message})
            history.append({"role": "assistant", "content": error_msg})
            stats = "Error occurred"

        return history, history, stats, "", "", ""

chat_instance = AgentWorldChat()

# ===== UI =====
CSS = """
.gr-accordion { border: 1px solid #e5e7eb !important; border-radius: 8px !important; }
"""

with gr.Blocks(
    title="Qwen-AgentWorld-35B-A3B - World Model Simulator",
) as demo:
    gr.Markdown("""
    # Qwen-AgentWorld-35B-A3B — Agent 环境模拟器

    一个模型覆盖 **7 个 Agent 交互领域**。
    35B MoE（3B 激活）| 262K 上下文 | Apache 2.0

    模型：[Qwen/Qwen-AgentWorld-35B-A3B](https://huggingface.co/Qwen/Qwen-AgentWorld-35B-A3B)
    """)

    with gr.Row():
        with gr.Column(scale=1):
            domain_selector = gr.Dropdown(
                choices=list(DOMAIN_PROMPTS.keys()),
                value="Terminal",
                label="模拟领域",
                info="选择 Agent 交互领域",
                interactive=True,
            )

            system_prompt_box = gr.Textbox(
                label="系统提示词（自动按领域设置）",
                value=get_system_prompt("Terminal"),
                lines=6,
                max_lines=10,
                interactive=False,
            )

            with gr.Accordion("生成参数（推荐默认值）", open=False):
                temperature_slider = gr.Slider(
                    0.0, 2.0, 0.6, step=0.05,
                    label="温度",
                    info="默认: 0.6",
                )
                top_p_slider = gr.Slider(
                    0.1, 1.0, 0.95, step=0.05,
                    label="Top-P",
                    info="默认: 0.95",
                )
                top_k_slider = gr.Slider(
                    1, 100, 20, step=1,
                    label="Top-K",
                    info="默认: 20",
                )
                max_tokens_slider = gr.Slider(
                    512, 32768, 4096, step=512,
                    label="最大生成 Token 数",
                    info="长轨迹模拟最多 32768",
                )

            reset_btn = gr.Button("清空对话", variant="secondary", size="sm")
            stats_box = gr.Textbox(label="推理统计", value="", interactive=False, max_lines=1)

        with gr.Column(scale=3):
            gr.Markdown("### 模拟环境")
            chatbot = gr.Chatbot(label="环境状态输出", height=500)
            with gr.Row():
                msg_input = gr.Textbox(
                    label="动作 / 命令",
                    placeholder="示例：Action: execute_bash\nCommand: ls -la /home/user/",
                    lines=3,
                    scale=4,
                )
                send_btn = gr.Button("执行", variant="primary", size="lg", scale=1, min_width=120)

    state = gr.State([])
    dummy = gr.State("")

    # Domain change -> update system prompt
    def on_domain_change(domain_key):
        prompt = chat_instance.set_domain(domain_key)
        return prompt, "", ""

    domain_selector.change(
        fn=on_domain_change,
        inputs=[domain_selector],
        outputs=[system_prompt_box, msg_input, stats_box],
    )

    # Send
    send_btn.click(
        fn=chat_instance.chat,
        inputs=[msg_input, state, temperature_slider, top_p_slider, top_k_slider, max_tokens_slider],
        outputs=[chatbot, state, stats_box, msg_input, msg_input, msg_input],
    )

    msg_input.submit(
        fn=chat_instance.chat,
        inputs=[msg_input, state, temperature_slider, top_p_slider, top_k_slider, max_tokens_slider],
        outputs=[chatbot, state, stats_box, msg_input, msg_input, msg_input],
    )

    reset_btn.click(
        fn=chat_instance.reset,
        inputs=[],
        outputs=[chatbot, state],
    ).then(fn=lambda: "", outputs=[stats_box])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False,
                theme=gr.themes.Soft(), css=CSS)
