#!/usr/bin/env python3
"""LocateAnything-3B Gradio WebUI — 视觉定位/目标检测"""

import os
import re
import time
import threading
from pathlib import Path
from urllib.request import urlopen

import gradio as gr
import torch
from PIL import Image, ImageDraw
from transformers import AutoModel, AutoTokenizer, AutoProcessor

MODEL_PATH = os.environ.get("MODEL_PATH", "/gemini/pretrain/LocateAnything-3B-model")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32

class LocateAnythingWorker:
    """Stateful worker — loads model once, serves perception queries."""

    def __init__(self, model_path: str, device: str = DEVICE, dtype=DTYPE):
        self.device = device
        self.dtype = dtype

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=dtype,
            trust_remote_code=True,
            local_files_only=True,
        ).to(device).eval()

    @torch.no_grad()
    def predict(
        self,
        image: Image.Image,
        question: str,
        generation_mode: str = "hybrid",
        max_new_tokens: int = 2048,
        temperature: float = 0.7,
        verbose: bool = False,
    ) -> dict:
        messages = [
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ]}
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text], images=[image], return_tensors="pt"
        ).to(self.device)

        pixel_values = inputs["pixel_values"].to(self.dtype)
        input_ids = inputs["input_ids"]
        image_grid_hws = inputs.get("image_grid_hws", None)

        response = self.model.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=inputs["attention_mask"],
            image_grid_hws=image_grid_hws,
            tokenizer=self.tokenizer,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            generation_mode=generation_mode,
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.1,
            verbose=verbose,
        )

        result = {"answer": response[0] if isinstance(response, tuple) else response}
        if isinstance(response, tuple) and len(response) >= 3:
            result["history"] = response[1]
            result["stats"] = response[2]
        return result

    def detect(self, image: Image.Image, categories: list[str], **kwargs) -> dict:
        cats = "</c>".join(categories)
        prompt = f"Locate all the instances that matches the following description: {cats}."
        return self.predict(image, prompt, **kwargs)

    def ground_single(self, image: Image.Image, phrase: str, **kwargs) -> dict:
        prompt = f"Locate a single instance that matches the following description: {phrase}."
        return self.predict(image, prompt, **kwargs)

    def ground_multi(self, image: Image.Image, phrase: str, **kwargs) -> dict:
        prompt = f"Locate all the instances that match the following description: {phrase}."
        return self.predict(image, prompt, **kwargs)

    def detect_text(self, image: Image.Image, **kwargs) -> dict:
        prompt = "Detect all the text in box format."
        return self.predict(image, prompt, **kwargs)

    def point(self, image: Image.Image, phrase: str, **kwargs) -> dict:
        prompt = f"Point to: {phrase}."
        return self.predict(image, prompt, **kwargs)

    @staticmethod
    def parse_boxes(answer: str, image_width: int, image_height: int) -> list[dict]:
        boxes = []
        for m in re.finditer(r"<box><(\d+)><(\d+)><(\d+)><(\d+)></box>", answer):
            x1, y1, x2, y2 = [int(g) for g in m.groups()]
            boxes.append({
                "x1": x1 / 1000 * image_width,
                "y1": y1 / 1000 * image_height,
                "x2": x2 / 1000 * image_width,
                "y2": y2 / 1000 * image_height,
            })
        return boxes

    @staticmethod
    def parse_points(answer: str, image_width: int, image_height: int) -> list[dict]:
        points = []
        for m in re.finditer(r"<box><(\d+)><(\d+)></box>", answer):
            x, y = int(m.group(1)), int(m.group(2))
            points.append({
                "x": x / 1000 * image_width,
                "y": y / 1000 * image_height,
            })
        return points

worker: LocateAnythingWorker | None = None
MODEL_LOADED = threading.Event()

def load_model_background():
    """Background thread: loads model, sets MODEL_LOADED flag."""
    global worker
    print("[background] Loading model...", flush=True)
    worker = LocateAnythingWorker(MODEL_PATH)
    MODEL_LOADED.set()
    print(f"[background] Model loaded! Device: {worker.device}", flush=True)

def run_inference(image, prompt_text, mode, max_tokens, temperature):
    if worker is None:
        return image, "Model not loaded yet, please wait..."

    result = worker.predict(
        image, prompt_text,
        generation_mode=mode,
        max_new_tokens=int(max_tokens),
        temperature=temperature,
    )

    raw = result["answer"]
    if isinstance(raw, torch.Tensor):
        raw = worker.tokenizer.decode(raw[0], skip_special_tokens=False)

    # Parse boxes & draw
    w, h = image.size
    boxes = worker.parse_boxes(raw, w, h)
    points = worker.parse_points(raw, w, h)

    draw = image.copy().convert("RGB")
    draw_img = ImageDraw.Draw(draw)

    for i, b in enumerate(boxes):
        draw_img.rectangle([b["x1"], b["y1"], b["x2"], b["y2"]], outline="red", width=3)
        draw_img.text((b["x1"], b["y1"]), str(i + 1), fill="red")

    for p in points:
        r = 5
        draw_img.ellipse([p["x"] - r, p["y"] - r, p["x"] + r, p["y"] + r], fill="blue")

    stats = result.get("stats", None)
    if stats:
        raw += f"\n\n[Stats] {stats}"

    return draw, raw.strip()

CSS = """
.gr-box {border-radius: 8px;}
h1 {text-align: center;}
"""

with gr.Blocks(title="LocateAnything-3B", css=CSS) as demo:
    gr.Markdown("# 🎯 LocateAnything-3B")
    gr.Markdown("NVIDIA 视觉定位模型 — 支持目标检测、短语定位、文字检测、GUI 定位、点定位")

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="pil", label="上传图片")
            prompt_input = gr.Textbox(
                label="提示词",
                placeholder='例如: "Locate all the instances that matches the following description: person.</c>car."',
                lines=3,
            )
            with gr.Row():
                mode_input = gr.Dropdown(
                    choices=["hybrid", "fast", "slow"],
                    value="hybrid",
                    label="生成模式",
                )
                temp_input = gr.Slider(0.1, 1.0, 0.7, step=0.05, label="温度")
                tokens_input = gr.Slider(256, 8192, 2048, step=256, label="最大 Token")

            submit_btn = gr.Button("运行推理", variant="primary")

            gr.Markdown("### 快捷模板")
            gr.Examples(
                examples=[
                    ["Locate all the instances that matches the following description: person.</c>car.</c>bicycle."],
                    ["Locate all the instances that match the following description: people wearing red shirts."],
                    ["Detect all the text in box format."],
                    ["Point to: the traffic light."],
                ],
                inputs=prompt_input,
            )

        with gr.Column(scale=1):
            image_output = gr.Image(type="pil", label="检测结果")
            text_output = gr.Textbox(label="模型输出", lines=10, max_lines=20)

    submit_btn.click(
        fn=run_inference,
        inputs=[image_input, prompt_input, mode_input, tokens_input, temp_input],
        outputs=[image_output, text_output],
    )

    gr.Markdown("---")
    gr.Markdown(
        f"> 模型路径: `{MODEL_PATH}` | 设备: `{DEVICE}` | 精度: `{DTYPE}`\n"
        f"> 加载状态: {'✅ 已加载' if MODEL_LOADED.is_set() else '⏳ 加载中...'}"
    )

if __name__ == "__main__":
    # Start model loading in background (UI starts immediately for health check)
    threading.Thread(target=load_model_background, daemon=True).start()

    # Keep-Alive thread (prevent idle timeout)
    def keep_alive():
        while True:
            time.sleep(300)  # 5 min
            try:
                urlopen("http://127.0.0.1:7860", timeout=10)
            except Exception:
                pass

    threading.Thread(target=keep_alive, daemon=True).start()

    demo.launch(server_name="0.0.0.0", server_port=7860)
