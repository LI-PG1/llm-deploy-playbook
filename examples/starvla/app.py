#!/usr/bin/env python3
"""
StarVLA QwenOFT - VLA Robot Action Inference Service
=====================================================
Gradio WebUI for predicting robot actions from image + instruction.

Model: Qwen3-VL-4B-Instruct + MLP Action Head (QwenOFT)
Checkpoint: StarVLA/Qwen3-VL-OFT-LIBERO-4in1 (LIBERO 4in1)
"""

import os, sys, threading, time, io
import numpy as np
from PIL import Image
from pathlib import Path

# Ensure starVLA source is importable
CODE_DIR = "/gemini/code"
if os.path.isdir(CODE_DIR):
    sys.path.insert(0, CODE_DIR)
else:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "source"))

import gradio as gr
import torch

# Paths
VLM_PATH = os.environ.get("VLM_PATH", "/gemini/pretrain/models/Qwen3-VL-4B-Instruct")
CKPT_PATH = os.environ.get(
    "CKPT_PATH",
    "/gemini/pretrain/models/StarVLA/Qwen3-VL-OFT-LIBERO-4in1/checkpoints/steps_50000_pytorch_model.pt"
)

# Action dimension labels for LIBERO (7-DoF)
ACTION_LABELS = [
    "x (前后)", "y (左右)", "z (上下)",
    "roll (翻滚)", "pitch (俯仰)", "yaw (偏航)",
    "gripper (夹爪)",
]

# Global model handle
_model = None
_model_loaded = False
_model_error = None

# =========================================================================
#  Model Loading
# =========================================================================

def load_model():
    """Load StarVLA QwenOFT model. Runs once at startup."""
    global _model, _model_loaded, _model_error
    try:
        from omegaconf import OmegaConf
        from starVLA.model.framework.share_tools import read_mode_config, dict_to_namespace
        from starVLA.model.framework.base_framework import build_framework

        print(f"[load] VLM:       {VLM_PATH}")
        print(f"[load] Checkpoint: {CKPT_PATH}")

        # 1. Load config from checkpoint
        model_config, norm_stats = read_mode_config(CKPT_PATH)

        # 2. Override base_vlm path for platform deployment
        model_config["framework"]["qwenvl"]["base_vlm"] = VLM_PATH
        model_config["framework"]["qwenvl"]["attn_implementation"] = "flash_attention_2"

        # 3. Convert to namespace and build model
        config = dict_to_namespace(model_config)
        model = build_framework(config)

        # 4. Load weights
        print("[load] Loading state_dict...")
        state_dict = torch.load(CKPT_PATH, map_location="cpu")
        model.load_state_dict(state_dict, strict=True)
        model.norm_stats = norm_stats

        # 5. Move to GPU
        print("[load] Moving to GPU...")
        model = model.to("cuda", dtype=torch.bfloat16).eval()
        torch.cuda.empty_cache()

        _model = model
        _model_loaded = True
        print(f"[load] Done. GPU memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

    except Exception as e:
        _model_error = str(e)
        print(f"[load] FAILED: {e}")
        import traceback
        traceback.print_exc()

# Background loading thread
_thread = threading.Thread(target=load_model, daemon=True)
_thread.start()

# =========================================================================
#  Inference
# =========================================================================

def predict(image: np.ndarray, instruction: str, progress=gr.Progress()):
    """Run inference: image + instruction -> predicted actions.

    Args:
        image: numpy array (H, W, 3), uint8
        instruction: natural language task description

    Returns:
        table_html: HTML table of action predictions
        plot_path: line chart path
        raw_text: raw numpy dump
    """
    global _model, _model_loaded, _model_error

    if _model_error:
        return f"<div style='color:red; font-size:18px;'>模型加载失败: {_model_error}</div>", None, ""

    if not _model_loaded:
        return "<div style='color:orange; font-size:18px;'>模型正在加载，请稍候...</div>", None, ""

    progress(0, desc="Preparing input...")

    # Validate image
    if image is None:
        return "<div style='color:red;'>Please upload an image.</div>", None, ""

    pil_img = Image.fromarray(image.astype(np.uint8)).convert("RGB")

    # Build example dict
    example = {
        "image": [pil_img],
        "lang": instruction.strip() or "pick up the object",
    }

    progress(0.3, desc="Running VLA inference...")

    try:
        with torch.inference_mode():
            output = _model.predict_action(examples=[example])
        normalized_actions = output["normalized_actions"]  # (1, T, D)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"<div style='color:red;'>推理错误: {str(e)}</div>", None, ""

    progress(0.7, desc="生成可视化图表...")

    actions = normalized_actions[0]  # (T, 7)
    T, D = actions.shape

    # --- HTML Table ---
    rows = []
    rows.append("<table style='border-collapse:collapse; width:100%; font-size:14px;'>")
    # Header
    rows.append("<tr style='background:#4a90d9; color:white;'>")
    rows.append("<th style='padding:8px; border:1px solid #ddd;'>Step</th>")
    for label in ACTION_LABELS:
        rows.append(f"<th style='padding:8px; border:1px solid #ddd;'>{label}</th>")
    rows.append("</tr>")
    # Data rows
    for t in range(T):
        bg = "#f8f9fa" if t % 2 == 0 else "white"
        rows.append(f"<tr style='background:{bg};'>")
        rows.append(f"<td style='padding:6px; border:1px solid #ddd; text-align:center;'>{t + 1}</td>")
        for d in range(D):
            val = actions[t, d]
            color = "#e74c3c" if abs(val) > 0.5 else "#2ecc71" if val > 0 else "#95a5a6"
            rows.append(
                f"<td style='padding:6px; border:1px solid #ddd; text-align:center; "
                f"color:{color}; font-weight:bold;'>{val:.4f}</td>"
            )
        rows.append("</tr>")
    rows.append("</table>")
    table_html = "".join(rows)

    # --- Matplotlib Line Chart ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(D, 1, figsize=(10, 1.5 * D + 1), sharex=True)
        fig.suptitle("预测动作 (7-DoF, 归一化)", fontsize=14, y=1.02)

        time_steps = np.arange(T)

        for d in range(D):
            ax = axes[d]
            ax.plot(time_steps, actions[:, d], "o-", color="#3498db", linewidth=2, markersize=5)
            ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
            ax.set_ylabel(ACTION_LABELS[d], fontsize=10)
            ax.set_ylim(-1.1, 1.1)
            ax.grid(True, alpha=0.3)
            if d == D - 1:
                ax.set_xlabel("动作步数", fontsize=10)

        plt.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
        buf.seek(0)
        plt.close(fig)
        plot_img = Image.open(buf)
    except Exception as e:
        print(f"[plot] WARNING: matplotlib failed ({e})")
        plot_img = None

    # --- Raw text ---
    raw_text = np.array2string(actions, precision=4, suppress_small=True)

    progress(1.0, desc="完成！")

    return table_html, plot_img, raw_text


# =========================================================================
#  Gradio UI
# =========================================================================

TITLE = "StarVLA QwenOFT - VLA 机器人动作预测服务"
DESCRIPTION = """
上传摄像头图像并用自然语言描述任务，模型将预测 7-DoF 机器人动作轨迹。

**模型**: Qwen3-VL-4B-Instruct + MLP 动作头 (QwenOFT)  
**检查点**: StarVLA/Qwen3-VL-OFT-LIBERO-4in1 (LIBERO 4in1)  
**动作空间**: x, y, z, roll, pitch, yaw, gripper（归一化至 [-1, 1]）
"""

with gr.Blocks(title=TITLE) as demo:
    gr.Markdown(f"# {TITLE}")
    gr.Markdown(DESCRIPTION)

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(
                label="摄像头图像",
                type="numpy",
                height=400,
            )
            text_input = gr.Textbox(
                label="任务指令",
                placeholder='例如："pick up the red block" 或 "open the drawer"',
                value="pick up the red block",
            )
            predict_btn = gr.Button("预测动作", variant="primary", size="lg")

        with gr.Column(scale=1):
            status_box = gr.HTML(
                value="<div style='color:gray;'>正在加载模型，请稍候...</div>",
                label="状态",
            )
            with gr.Tabs():
                with gr.TabItem("数据表格"):
                    table_output = gr.HTML(label="动作预测结果")
                with gr.TabItem("趋势图"):
                    plot_output = gr.Image(label="动作轨迹", height=400)
                with gr.TabItem("原始数据"):
                    raw_output = gr.Textbox(label="原始动作数据 (numpy)", lines=10)

    # Quick examples
    gr.Markdown("### 快速示例")
    examples_data = [
        ["pick up the red block"],
        ["open the drawer"],
        ["push the button"],
        ["place the object on the table"],
    ]
    gr.Examples(
        examples=examples_data,
        inputs=[text_input],
        label="点击尝试：",
    )

    # Periodic status refresh (every 2s)
    def get_status():
        if _model_error:
            return f"<div style='color:red; font-weight:bold;'>模型加载失败: {_model_error}</div>"
        elif _model_loaded:
            return "<div style='color:green; font-weight:bold;'>模型已就绪</div>"
        else:
            return "<div style='color:orange;'>正在加载模型...</div>"

    status_box.value = get_status()
    timer = gr.Timer(value=2)
    timer.tick(get_status, outputs=status_box)

    # Predict
    predict_btn.click(
        fn=predict,
        inputs=[image_input, text_input],
        outputs=[table_output, plot_output, raw_output],
    )

# =========================================================================
#  Launch
# =========================================================================

if __name__ == "__main__":
    PORT = int(os.environ.get("GRADIO_SERVER_PORT", 7860))

    print(f"Starting Gradio on 0.0.0.0:{PORT}")
    demo.launch(
        server_name="0.0.0.0",
        server_port=PORT,
        quiet=False,
        theme="soft",
    )
