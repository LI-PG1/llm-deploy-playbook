import gradio as gr
import os
import torch
from PIL import Image
from threading import Lock

# check_model_inputs 兼容 transformers 4.57
import transformers.utils.generic as _G
_orig = _G.check_model_inputs
def _patched(func=None):
    if func is None:
        return _orig
    return _orig(func)
_G.check_model_inputs = _patched

MODEL_PATH = os.environ.get("MODEL_PATH", "/gemini/pretrain/Keye-VL-2.0-30B-A3B")

MODEL = None
PROCESSOR = None
MODEL_LOCK = Lock()

def load_model():
    global MODEL, PROCESSOR
    if MODEL is not None:
        return MODEL, PROCESSOR

    with MODEL_LOCK:
        if MODEL is not None:
            return MODEL, PROCESSOR

        from transformers import AutoModel, AutoProcessor

        print("[INFO] Loading model from", MODEL_PATH)

        MODEL = AutoModel.from_pretrained(
            MODEL_PATH,
            dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            local_files_only=True,
        )

        PROCESSOR = AutoProcessor.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True,
            local_files_only=True,
        )

        print("[INFO] Model loaded, device_map:", MODEL.hf_device_map)
        return MODEL, PROCESSOR

def chat_with_image(image, text, history):
    if image is None and not text:
        history.append({"role": "assistant", "content": "请上传图片并输入问题"})
        return history

    if not text or not text.strip():
        history.append({"role": "assistant", "content": "请输入问题"})
        return history

    model, processor = load_model()

    pil_image = Image.fromarray(image) if image is not None and not isinstance(image, Image.Image) else image

    messages = []
    if pil_image is not None:
        messages.append({
            "role": "user",
            "content": [
                {"type": "image", "image": pil_image},
                {"type": "text", "text": text.strip()},
            ],
        })
    else:
        messages.append({"role": "user", "content": [{"type": "text", "text": text.strip()}]})

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )

    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )

    input_len = inputs["input_ids"].shape[1]
    output_ids = generated_ids[0][input_len:]
    response = processor.decode(output_ids, skip_special_tokens=True)

    history.append({"role": "user", "content": text})
    history.append({"role": "assistant", "content": response})
    return history

def clear_chat():
    return []

with gr.Blocks(title="Keye-VL-2.0 多模态对话") as demo:
    gr.Markdown("# Keye-VL-2.0-30B-A3B 多模态视觉对话")
    gr.Markdown("快手 Keye-VL-2.0 MoE · 30B总参/3B激活 · 上传图片输入问题")

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="numpy", label="上传图片")
            clear_btn = gr.Button("清空对话", variant="secondary")

        with gr.Column(scale=2):
            chatbot = gr.Chatbot(label="对话", height=500)
            text_input = gr.Textbox(
                placeholder="输入你的问题...",
                label="问题",
                lines=2,
            )
            submit_btn = gr.Button("发送", variant="primary")

    submit_btn.click(
        fn=chat_with_image,
        inputs=[image_input, text_input, chatbot],
        outputs=chatbot,
    ).then(lambda: "", None, text_input)

    text_input.submit(
        fn=chat_with_image,
        inputs=[image_input, text_input, chatbot],
        outputs=chatbot,
    ).then(lambda: "", None, text_input)

    clear_btn.click(fn=clear_chat, outputs=chatbot)

if __name__ == "__main__":
    demo.queue(max_size=10).launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )
