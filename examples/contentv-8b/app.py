import os
import time
import threading
import tempfile
import uuid
import numpy as np
import gradio as gr
import torch
from diffusers.utils import export_to_video
from diffusers import AutoencoderKLWan
from contentv_transformer import SD3Transformer3DModel
from contentv_pipeline import ContentVPipeline

MODEL_PATH = os.environ.get("MODEL_PATH", "/gemini/pretrain/ContentV-8B")

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_SSL_VERIFY"] = "1"

DEFAULT_PROMPT = "A young musician sits on a rustic wooden stool in a cozy, dimly lit room, strumming an acoustic guitar with a worn, sunburst finish."
DEFAULT_NEGATIVE = "overexposed, low quality, deformation, a poor composition, bad hands, bad teeth, bad eyes, bad limbs, distortion"

pipe = None
MODEL_LOCK = threading.Lock()


def load_model():
    global pipe
    if pipe is not None:
        return pipe

    with MODEL_LOCK:
        if pipe is not None:
            return pipe

        print(f"[LOAD] Loading ContentV-8B from {MODEL_PATH}...")
        t0 = time.time()

        print("  [1/3] Loading VAE (FP32)...")
        vae = AutoencoderKLWan.from_pretrained(
            MODEL_PATH,
            subfolder="vae",
            torch_dtype=torch.float32,
            local_files_only=True,
        )

        print("  [2/3] Loading Transformer (BF16)...")
        transformer = SD3Transformer3DModel.from_pretrained(
            MODEL_PATH,
            subfolder="transformer",
            torch_dtype=torch.bfloat16,
            local_files_only=True,
        )

        print("  [3/3] Loading Pipeline...")
        pipe = ContentVPipeline.from_pretrained(
            MODEL_PATH,
            vae=vae,
            transformer=transformer,
            torch_dtype=torch.bfloat16,
            local_files_only=True,
        )

        if torch.cuda.is_available():
            print(f"  Moving to CUDA ({torch.cuda.get_device_name(0)})...")
        pipe.to("cuda")

        t1 = time.time()
        print(f"[LOAD] Model loaded in {t1 - t0:.1f}s")
        return pipe


def generate_video(
    prompt,
    negative_prompt,
    width,
    height,
    num_frames,
    num_inference_steps,
    guidance_scale,
    seed,
    progress=gr.Progress(track_tqdm=True),
):
    try:
        pipe = load_model()

        generator = torch.Generator(device="cuda")
        if seed >= 0:
            generator = generator.manual_seed(seed)

        print(f"[GEN] prompt='{prompt[:60]}...' | frames={num_frames} | "
              f"size={width}x{height} | steps={num_inference_steps} | "
              f"guidance={guidance_scale} | seed={seed}")

        progress(0.1, desc="Loading model...")

        output = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt or DEFAULT_NEGATIVE,
            width=width,
            height=height,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )

        frames = output.frames[0]

        # export_to_video 要求 ndarray，转 (F, H, W, C) uint8
        if isinstance(frames, torch.Tensor):
            frames = (frames * 255).clamp(0, 255).cpu().numpy().astype(np.uint8)
        elif frames.dtype == np.float32 or frames.dtype == np.float64:
            frames = (frames * 255).clip(0, 255).astype(np.uint8)

        out_dir = tempfile.mkdtemp()
        out_path = os.path.join(out_dir, f"contentv_{uuid.uuid4().hex[:8]}.mp4")

        progress(0.95, desc="Exporting video...")
        export_to_video(frames, out_path, fps=24)

        print(f"[GEN] Done: {out_path}")
        return out_path

    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
        raise gr.Error(f"Generation failed: {e}")


def warmup():
    """Warm up the model at startup."""
    if not os.environ.get("SKIP_WARMUP"):
        print("[WARMUP] Starting model preload...")
        try:
            load_model()
        except Exception as e:
            print(f"[WARMUP] Preload failed (will retry on first request): {e}")


# ---- Gradio UI ----

with gr.Blocks(
    title="ContentV-8B Video Generator",
    theme=gr.themes.Soft(),
) as demo:
    gr.Markdown(
        """
        # ContentV-8B 视频生成

        ByteDance 开源的文生视频扩散模型（基于 SD3.5 Large + Wan-VAE）
        """
    )

    with gr.Row():
        with gr.Column(scale=2):
            prompt = gr.Textbox(
                label="Prompt",
                placeholder="Describe the video you want to generate...",
                value=DEFAULT_PROMPT,
                lines=3,
            )
            negative_prompt = gr.Textbox(
                label="Negative Prompt",
                placeholder="Things to avoid...",
                value=DEFAULT_NEGATIVE,
                lines=2,
            )

        with gr.Column(scale=1):
            with gr.Group():
                width = gr.Slider(256, 1024, value=768, step=64, label="Width")
                height = gr.Slider(256, 768, value=432, step=64, label="Height")
                num_frames = gr.Slider(25, 201, value=125, step=4, label="Frames (~FPS*秒)")

            with gr.Group():
                num_inference_steps = gr.Slider(10, 100, value=50, step=1, label="Inference Steps")
                guidance_scale = gr.Slider(1.0, 15.0, value=6.0, step=0.5, label="Guidance Scale")
                seed = gr.Slider(-1, 999999, value=-1, step=1, label="Seed (-1 = random)")

            generate_btn = gr.Button("Generate Video", variant="primary", size="lg")

    with gr.Row():
        video_output = gr.Video(label="Generated Video", height=480)

    generate_btn.click(
        fn=generate_video,
        inputs=[
            prompt,
            negative_prompt,
            width,
            height,
            num_frames,
            num_inference_steps,
            guidance_scale,
            seed,
        ],
        outputs=video_output,
    )

    gr.Markdown(
        """
        ---
        **Tips:**
        - 默认分辨率 768×432，帧数 125 ≈ 5秒 @ 24fps
        - 视频生成耗时约 2-5 分钟（80GB 显存），取决于帧数和步数
        - 非等比分辨率会自动居中裁剪
        - 增大步数可提升质量，但生成时间更长
        """
    )

if __name__ == "__main__":
    # Preload model in background thread
    bg_thread = threading.Thread(target=warmup, daemon=True)
    bg_thread.start()

    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
    )
