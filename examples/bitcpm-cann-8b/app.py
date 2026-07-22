"""
BitCPM-CANN-8B Gradio WebUI (gradio 4.x compatible)
"""
import gradio as gr
import os
import torch
from threading import Lock

MODEL_PATH = os.environ.get("MODEL_PATH", "/gemini/pretrain/BitCPM-CANN-8B")
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

        print(f"[INFO] Loading model from {MODEL_PATH}", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH, trust_remote_code=True, local_files_only=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH, torch_dtype=torch.bfloat16,
            device_map="auto", trust_remote_code=True, local_files_only=True,
        )
        MODEL = model
        TOKENIZER = tokenizer
        print(f"[INFO] Model loaded", flush=True)
        return MODEL, TOKENIZER


def generate_response(prompt):
    """Benchmark 测试接口 — 接收 prompt 返回生成的文本"""
    model, tokenizer = load_model()
    responds, _ = model.chat(
        tokenizer, prompt,
        temperature=0.7, top_p=0.7, max_new_tokens=2048,
    )
    return responds


def chat(message, history):
    if not message or not message.strip():
        return "", history

    model, tokenizer = load_model()

    try:
        responds, new_history = model.chat(
            tokenizer, message.strip(),
            temperature=0.7, top_p=0.7, max_new_tokens=2048,
        )
        history.append((message, responds))
        return "", history
    except Exception as e:
        import traceback
        print(traceback.format_exc(), flush=True)
        history.append((message, f"Error: {str(e)}"))
        return "", history


def clear_chat():
    return [], ""


with gr.Blocks(title="BitCPM-CANN-8B") as demo:
    gr.Markdown("""
    # BitCPM-CANN-8B 对话
    ### 面壁智能 OpenBMB - 1.58-bit 三值量化端侧大模型
    """)

    chatbot = gr.Chatbot(label="对话", height=500)

    with gr.Row():
        msg = gr.Textbox(placeholder="输入你的问题...", label="消息", scale=4, container=False)
        send_btn = gr.Button("发送", variant="primary", scale=1)
        clear_btn = gr.Button("清空", scale=1)

    gr.Markdown("**提示:** 1.58-bit 三值量化，约 6 倍显存节省。支持中英文对话。")

    send_btn.click(chat, inputs=[msg, chatbot], outputs=[msg, chatbot])
    msg.submit(chat, inputs=[msg, chatbot], outputs=[msg, chatbot])
    clear_btn.click(clear_chat, outputs=[chatbot, msg])

if __name__ == "__main__":
    import logging, warnings
    logging.getLogger("uvicorn.error").disabled = True
    logging.getLogger("uvicorn.access").disabled = True
    warnings.filterwarnings("ignore")
    demo.queue(max_size=20).launch(server_name="0.0.0.0", server_port=7860, share=False, show_error=True)
