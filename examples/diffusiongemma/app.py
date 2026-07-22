"""
DiffusionGemma 26B-A4B-it — Gradio WebUI
Google DeepMind 扩散视觉语言 MoE 模型
支持文本对话 + 图像理解 + 视频理解（多模态输入）
离散文本扩散（non-autoregressive），block-autoregressive 多画布采样

完全离线加载，所有文件本地读取
"""
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import gradio as gr
import torch
from PIL import Image
from threading import Lock

MODEL_PATH = os.environ.get("MODEL_PATH", "/gemini/pretrain/diffusiongemma-26B-A4B-it")
# 候选搜索路径（适用于本地开发环境 vs 平台挂载）
_SEARCH_CANDIDATES = [
    MODEL_PATH,
    "/gemini/pretrain/diffusiongemma-26B-A4B-it",
    "/gemini/data-1/diffusiongemma-26B-A4B-it",
]

MODEL = None
PROCESSOR = None
MODEL_LOCK = Lock()

def find_model_path() -> str:
    """自动搜索模型目录"""
    for p in _SEARCH_CANDIDATES:
        cfg = os.path.join(p, "config.json")
        if os.path.isfile(cfg):
            print(f"[FIND] Model found: {p}")
            return p
    # fallback
    print(f"[FIND] Using default: {MODEL_PATH}")
    return MODEL_PATH

def load_model():
    global MODEL, PROCESSOR
    if MODEL is not None:
        return MODEL, PROCESSOR

    with MODEL_LOCK:
        if MODEL is not None:
            return MODEL, PROCESSOR

        import transformers

        print(f"[INFO] transformers version: {transformers.__version__}")

        try:
            from transformers import DiffusionGemmaForBlockDiffusion, AutoProcessor
        except ImportError:
            raise ImportError(
                f"transformers {transformers.__version__} does not have DiffusionGemmaForBlockDiffusion. "
                "This model requires transformers >= 5.8.0. "
                "Run: pip install -U transformers"
            )

        mp = find_model_path()
        print(f"[INFO] Loading model from {mp}")

        # 检查是否有 bf16 支持
        dt = torch.bfloat16 if torch.cuda.is_available() else torch.float32

        print("[INFO] Loading processor...")
        PROCESSOR = AutoProcessor.from_pretrained(
            mp,
            local_files_only=True,
        )

        print("[INFO] Loading model (this may take a few minutes)...")
        print(f"[INFO] Using dtype={dt}, device='cuda:0'")
        MODEL = DiffusionGemmaForBlockDiffusion.from_pretrained(
            mp,
            torch_dtype=dt,
            device_map="cuda:0",
            local_files_only=True,
        )
        MODEL.eval()

        print(f"[INFO] Model loaded successfully!")
        if torch.cuda.is_available():
            vram_used = torch.cuda.max_memory_allocated() / 1e9
            print(f"[INFO] GPU VRAM used: {vram_used:.2f} GB")

        return MODEL, PROCESSOR

def clean_output(text: str) -> str:
    """清理 DiffusionGemma 输出：移除标签和填充 token"""
    import re
    # 移除 channel 块
    text = re.sub(r'<\|channel\|>.*?</channel\|>', '', text, flags=re.DOTALL)
    # 移除所有尖括号标签（包括 <pad>, <turn|>, <|endoftext|> 等）
    text = re.sub(r'<[^>]*>', '', text)
    # 清理多余空白
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def chat_fn(image, text, history, system_prompt, enable_thinking, max_new_tokens, temperature):
    """
    Gradio 聊天函数
    image: np.ndarray (来自 gr.Image type="numpy") 或 None
    text: str 用户输入
    history: list[dict] Gradio Chatbot dict 格式
    system_prompt: str
    enable_thinking: bool
    max_new_tokens: int
    temperature: float (扩散采样器使用内部 EntropyBound 温度调度，此参数仅供参考)
    """
    if not text.strip() and image is None:
        history.append({"role": "assistant", "content": "请输入文本或上传图片后提问。"})
        return history

    model, processor = load_model()

    # 组装内容
    content_parts = []

    # 如果上传了图片
    if image is not None:
        pil_image = Image.fromarray(image) if not isinstance(image, Image.Image) else image
        content_parts.append({"type": "image", "image": pil_image})

    # 用户文本
    if text.strip():
        content_parts.append({"type": "text", "text": text.strip()})

    # 构建完整消息列表（从 history 回溯 + system prompt + 当前输入）
    full_messages = []
    for h in history:
        if isinstance(h, dict):
            full_messages.append(h)

    if system_prompt.strip():
        think_tag = "<|think|>\n" if enable_thinking else ""
        sys_content = think_tag + system_prompt.strip()
        full_messages.insert(0, {"role": "system", "content": sys_content})
    elif enable_thinking:
        full_messages.insert(0, {"role": "system", "content": "<|think|>\n"})

    full_messages.append({"role": "user", "content": content_parts})

    try:
        # 应用 chat template
        inputs = processor.apply_chat_template(
            full_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        if torch.cuda.is_available():
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

        print(f"[INFER] Input tokens: {inputs['input_ids'].shape[-1]}")
        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        t0.record()

        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
            )
            generated_ids = generated.sequences  # DiffusionGemmaGenerationOutput -> tensor

        t1.record()
        torch.cuda.synchronize()
        elapsed = t0.elapsed_time(t1) / 1000.0

        input_len = inputs["input_ids"].shape[-1]
        output_ids = generated_ids[0][input_len:]
        raw_text = processor.decode(output_ids, skip_special_tokens=False)

        # 清理输出
        cleaned = clean_output(raw_text)

        print(f"[INFER] Output tokens: {output_ids.shape[-1]}, time: {elapsed:.1f}s, "
              f"speed: {output_ids.shape[-1] / elapsed:.1f} tok/s")

        if not cleaned.strip():
            cleaned = "(empty response — check thinking mode or generation config)"

        history.append({"role": "user", "content": text if text.strip() else "[图片]"})
        history.append({"role": "assistant", "content": cleaned})

    except Exception as e:
        import traceback
        err_msg = f"Error: {type(e).__name__}: {e}"
        print(f"[ERROR] {err_msg}")
        traceback.print_exc()
        history.append({"role": "user", "content": text if text.strip() else "[图片]"})
        history.append({"role": "assistant", "content": err_msg})

    return history

def clear_chat():
    return []

_TITLE = "DiffusionGemma 26B-A4B-it — 扩散视觉语言对话"
_DESC = """
**Google DeepMind** · [HuggingFace](https://huggingface.co/google/diffusiongemma-26B-A4B-it) · [Blog](https://blog.google/innovation-and-ai/technology/developers-tools/diffusion-gemma-faster-text-generation/)

25.2B 总参 / 3.8B 激活 · MoE 128专家(8 active) · 扩散文本生成 · 262K上下文 · Apache 2.0
"""
_FOOTNOTE = "DiffusionGemma 使用离散文本扩散（non-autoregressive），生成速度远快于因果语言模型。支持文本/图像/视频多模态输入。"

with gr.Blocks(title=_TITLE) as demo:
    gr.Markdown(f"# {_TITLE}")
    gr.Markdown(_DESC)

    with gr.Row():
        with gr.Column(scale=1):
            # 左侧：控制面板
            image_input = gr.Image(type="numpy", label="上传图片（可选）")

            with gr.Accordion("系统提示 / Thinking", open=True):
                system_prompt = gr.Textbox(
                    label="System Prompt",
                    placeholder="You are a helpful AI assistant...",
                    lines=3,
                )
                enable_thinking = gr.Checkbox(
                    label="启用 Thinking 模式（<|think|>）",
                    value=False,
                    info="开启后模型会先展示推理过程再输出答案",
                )

            with gr.Accordion("Generation 参数", open=False):
                max_tokens = gr.Slider(
                    minimum=256, maximum=4096, value=1024, step=256,
                    label="Max New Tokens (canvas=256)",
                    info="建议 256 的倍数",
                )
                temperature = gr.Slider(
                    minimum=0.0, maximum=2.0, value=0.7, step=0.1,
                    label="Temperature",
                )
                _gen_info = gr.Markdown(
                    "> Diffusion 采样器使用 Entropy-Bound Denoising，\n"
                    "> 温度在 0.8→0.4 线性衰减，最多 48 步去噪。"
                )

            clear_btn = gr.Button("清空对话", variant="secondary", size="sm")

        with gr.Column(scale=3):
            # 右侧：对话区域
            chatbot = gr.Chatbot(
                label="对话",
                height=600,
            )
            with gr.Row():
                text_input = gr.Textbox(
                    placeholder="输入你的问题（支持图片+文本混合输入）...",
                    label="输入",
                    lines=2,
                    scale=4,
                )
                submit_btn = gr.Button("发送", variant="primary", size="lg", scale=1)

    gr.Markdown(f"---\n{_FOOTNOTE}")

    # 事件绑定
    submit_btn.click(
        fn=chat_fn,
        inputs=[image_input, text_input, chatbot, system_prompt, enable_thinking, max_tokens, temperature],
        outputs=chatbot,
    ).then(lambda: "", None, text_input).then(lambda: None, None, image_input)

    text_input.submit(
        fn=chat_fn,
        inputs=[image_input, text_input, chatbot, system_prompt, enable_thinking, max_tokens, temperature],
        outputs=chatbot,
    ).then(lambda: "", None, text_input).then(lambda: None, None, image_input)

    clear_btn.click(fn=clear_chat, outputs=chatbot)

if __name__ == "__main__":
    print("=" * 60)
    print(f"  DiffusionGemma 26B-A4B-it — WebUI")
    mp = find_model_path()
    print(f"  Model: {mp}")
    print(f"  CUDA:  {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_total = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  GPU:   {gpu_name} | {vram_total:.1f} GB")
    print(f"  Debug: 加载模型（首次调用会耗时 1-5 分钟）...")
    print("=" * 60)

    # 后台预加载模型（不影响 Gradio 启动）
    import threading
    threading.Thread(target=load_model, daemon=True).start()

    server_name = os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0")
    server_port = int(os.environ.get("GRADIO_SERVER_PORT", "7860"))
    demo.queue(max_size=10).launch(
        server_name=server_name,
        server_port=server_port,
        theme=gr.themes.Soft(),
        share=False,
        show_error=True,
    )
