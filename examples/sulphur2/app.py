import gradio as gr
import os
import json
import time
import uuid
import urllib.request
import urllib.error
import socket

COMFYUI_URL = "http://127.0.0.1:8188"

# ================================================================
#  workflow format converter: ComfyUI GUI JSON → API prompt dict
# ================================================================

GUI_ONLY_TYPES = {"Reroute", "Note", "PrimitiveLinkFilter", "GroupNode", "GroupNodeSelector"}

_OBJECT_INFO_CACHE = None

def _get_object_info():
    """从 ComfyUI 获取所有节点的 input 定义（含默认值）。只调一次，缓存结果。"""
    global _OBJECT_INFO_CACHE
    if _OBJECT_INFO_CACHE is None:
        url = f"{COMFYUI_URL}/object_info"
        with urllib.request.urlopen(url, timeout=10) as resp:
            _OBJECT_INFO_CACHE = json.loads(resp.read().decode("utf-8"))
    return _OBJECT_INFO_CACHE

def workflow_to_api(workflow):
    """将 ComfyUI GUI workflow JSON 转为 API prompt 格式。"""
    # 1. 建立 node 索引
    nodes_by_id = {}
    for n in workflow.get("nodes", []):
        nodes_by_id[n.get("id")] = n

    if not nodes_by_id:
        raise RuntimeError("Workflow has no nodes")

    # 2. 追踪 Reroute 链路
    reroute_map = {}
    for link in workflow.get("links", []):
        if len(link) < 5:
            continue
        src_id, src_slot, dst_id, dst_slot = link[1], link[2], link[3], link[4]
        dst_node = nodes_by_id.get(dst_id)
        if dst_node and dst_node.get("type") == "Reroute":
            reroute_map[dst_id] = (src_id, src_slot)

    def resolve_src(src_id, src_slot):
        while src_id in reroute_map:
            src_id, src_slot = reroute_map[src_id]
        return src_id, src_slot

    # 3. 构建 prompt dict
    prompt = {}
    for n in workflow.get("nodes", []):
        ntype = n.get("type", "")
        if ntype in GUI_ONLY_TYPES:
            continue
        nid = str(n.get("id", ""))
        if not nid:
            continue

        inputs = {}
        wv = n.get("widgets_values") or []
        ni = n.get("inputs") or []

        # 只填充有 link 的 inputs（widgets_values 值可能对应不同名字，不直接用 index 映射）
        for inp in ni:
            name = inp.get("name", "")
            if name:
                inputs[name] = inp.get("default", None)

        prompt[nid] = {"class_type": ntype, "inputs": inputs}

    # 4. 填入连接关系
    for link in workflow.get("links", []):
        if len(link) < 5:
            continue
        src_id, src_slot, dst_id, dst_slot = link[1], link[2], link[3], link[4]

        dst_node = nodes_by_id.get(dst_id)
        if not dst_node or dst_node.get("type") in GUI_ONLY_TYPES:
            continue

        src_id, src_slot = resolve_src(src_id, src_slot)
        dst_inputs = dst_node.get("inputs") or []
        if dst_slot < len(dst_inputs):
            name = dst_inputs[dst_slot].get("name", "")
            if name and str(dst_id) in prompt:
                prompt[str(dst_id)]["inputs"][name] = [str(src_id), src_slot]

    # 5. 从 object_info 补全缺失的 required inputs + 填入 widget_values
    try:
        info = _get_object_info()
        for n in workflow.get("nodes", []):
            nid = str(n.get("id", ""))
            ntype = n.get("type", "")
            if ntype in GUI_ONLY_TYPES or nid not in prompt:
                continue
            node_info = info.get(ntype, {})
            if not node_info:
                continue

            required = node_info.get("input", {}).get("required", {})
            if not isinstance(required, dict):
                continue

            # widget_values 只对应非 link 的 required 输入——筛选出来按序映射
            wv = n.get("widgets_values") or []
            req_keys = list(required.keys())
            widget_keys = []
            for key in req_keys:
                existing = prompt[nid]["inputs"].get(key)
                if isinstance(existing, list):
                    continue          # 已被 link 填充
                if existing is not None and key in prompt[nid]["inputs"]:
                    continue          # 已有非 None 值
                widget_keys.append(key)

            for wi, key in enumerate(widget_keys):
                if wi < len(wv) and wv[wi] is not None:
                    prompt[nid]["inputs"][key] = wv[wi]
                else:
                    spec = required.get(key, [])
                    if isinstance(spec, list) and len(spec) > 1 and isinstance(spec[1], dict):
                        dv = spec[1].get("default")
                        if dv is not None:
                            prompt[nid]["inputs"][key] = dv
    except Exception:
        # object_info 不可用时退回到简单的 widget → input 映射
        for n in workflow.get("nodes", []):
            nid = str(n.get("id", ""))
            ntype = n.get("type", "")
            if ntype in GUI_ONLY_TYPES or nid not in prompt:
                continue
            wv = n.get("widgets_values") or []
            ni = n.get("inputs") or []
            for i, inp in enumerate(ni):
                name = inp.get("name", "")
                if name and i < len(wv) and wv[i] is not None:
                    if not isinstance(prompt[nid]["inputs"].get(name), list):
                        prompt[nid]["inputs"][name] = wv[i]

    return prompt

# ================================================================
#  workflow loading (local JSON only)
# ================================================================

def load_workflow_template(mode="t2v"):
    local_path = os.path.join(os.path.dirname(__file__), f"{mode}_base.json")
    if not os.path.exists(local_path):
        parent_path = os.path.join(os.path.dirname(__file__), "..", f"{mode}_base.json")
        if os.path.exists(parent_path):
            local_path = parent_path
        else:
            raise RuntimeError(f"Workflow JSON not found at {local_path}")
    with open(local_path, "r") as f:
        return json.load(f)

def find_node_by_type(workflow, node_type):
    for node in workflow["nodes"]:
        if node["type"] == node_type:
            return node
    return None

def find_nodes_by_type(workflow, node_type):
    return [n for n in workflow["nodes"] if n["type"] == node_type]

# ================================================================
#  ComfyUI API client
# ================================================================

def queue_prompt(prompt_workflow):
    api_prompt = workflow_to_api(prompt_workflow)
    payload = {"prompt": api_prompt, "extra_data": {"extra_pnginfo": {}}}
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(f"{COMFYUI_URL}/prompt", data=data)
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            if "error" in result:
                msg = json.dumps(result["error"]) if isinstance(result["error"], (dict, list)) else str(result["error"])
                raise RuntimeError(f"ComfyUI error: {msg}")
            return result
    except urllib.error.URLError as e:
        raise RuntimeError(f"ComfyUI not reachable: {e}")

def get_history(prompt_id):
    try:
        url = f"{COMFYUI_URL}/history/{prompt_id}"
        with urllib.request.urlopen(url, timeout=10) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError:
        return None

def wait_for_completion(prompt_id, timeout=300):
    start = time.time()
    while time.time() - start < timeout:
        history = get_history(prompt_id)
        if history and prompt_id in history:
            return history[prompt_id]
        time.sleep(2)
    raise TimeoutError("Generation timed out")

def get_output_files(history_entry):
    files = []
    for node_id, node_output in history_entry.get("outputs", {}).items():
        for output_name, output_data in node_output.items():
            for item in output_data:
                filename = item.get("filename", "")
                subfolder = item.get("subfolder", "")
                filetype = item.get("type", "")
                if filetype == "output":
                    if subfolder:
                        path = f"{COMFYUI_URL}/view?filename={filename}&subfolder={subfolder}&type=output"
                    else:
                        path = f"{COMFYUI_URL}/view?filename={filename}&type=output"
                    files.append((filename, path))
    return files

# ================================================================
#  Sulphur-2 inference engine
# ================================================================

class Sulphur2Inference:
    def __init__(self):
        self.output_dir = "/tmp/output"
        os.makedirs(self.output_dir, exist_ok=True)
        self._wait_for_comfyui()

    def _wait_for_comfyui(self, max_wait=60):
        start = time.time()
        while time.time() - start < max_wait:
            try:
                req = urllib.request.Request(f"{COMFYUI_URL}/system_stats")
                urllib.request.urlopen(req, timeout=5)
                return True
            except (urllib.error.URLError, socket.error):
                time.sleep(2)
        raise RuntimeError(f"ComfyUI did not start within {max_wait}s")

    def generate_video(self, prompt, negative_prompt, width, height, num_frames,
                       fps, steps, cfg_scale, seed, input_image=None):

        mode = "i2v" if input_image else "t2v"
        workflow = load_workflow_template(mode)

        prompt_nodes = find_nodes_by_type(workflow, "CLIPTextEncode")
        multiline_nodes = find_nodes_by_type(workflow, "PrimitiveStringMultiline")

        if len(prompt_nodes) >= 2:
            prompt_nodes[0]["widgets_values"] = [prompt]
            prompt_nodes[1]["widgets_values"] = [negative_prompt]

        for node in multiline_nodes:
            if node.get("title", "").lower() == "positive prompt":
                node["widgets_values"] = [prompt]
            elif node.get("title", "").lower() == "negative prompt":
                node["widgets_values"] = [negative_prompt]

        checkpoint_node = find_node_by_type(workflow, "CheckpointLoaderSimple")
        if checkpoint_node:
            checkpoint_node["widgets_values"] = ["sulphur_dev_bf16.safetensors"]

        vae_node = find_node_by_type(workflow, "VAELoader")
        if vae_node:
            vae_node["widgets_values"] = ["pixel_space"]

        latent_node = find_node_by_type(workflow, "EmptyLTXVLatentVideo")
        if latent_node:
            latent_node["widgets_values"] = [width, height, num_frames]

        int_nodes = find_nodes_by_type(workflow, "PrimitiveInt")
        for node in int_nodes:
            title = node.get("title", "").lower()
            if "seed" in title:
                node["widgets_values"] = [seed]
            elif "step" in title or title == "steps":
                node["widgets_values"] = [steps]
            elif "cfg" in title:
                node["widgets_values"] = [cfg_scale]
            elif "frame" in title or "fps" in title:
                node["widgets_values"] = [fps]

        sampler_node = find_node_by_type(workflow, "SamplerCustomAdvanced")
        if sampler_node:
            sampler_node["widgets_values"] = [cfg_scale]

        if input_image:
            image_node = find_node_by_type(workflow, "LoadImage")
            if image_node:
                image_node["widgets_values"] = [input_image]

        result = queue_prompt(workflow)
        prompt_id = result.get("prompt_id")
        if not prompt_id:
            raise RuntimeError("Failed to queue prompt in ComfyUI")

        history = wait_for_completion(prompt_id)

        output_files = get_output_files(history)
        video_files = [f for f in output_files if f[0].endswith(('.mp4', '.webm', '.gif', '.mov'))]
        if not video_files:
            img_files = [f for f in output_files if f[0].endswith(('.png', '.jpg', '.jpeg'))]
            return None, f"Generated {len(img_files)} image frames (no video)"

        best = sorted(video_files, key=lambda x: x[0])[-1]
        return best[1], None

inference = None

def get_inference():
    global inference
    if inference is None:
        inference = Sulphur2Inference()
    return inference

def generate_video_fn(prompt, negative_prompt, width, height, num_frames,
                      fps, steps, cfg_scale, seed, input_image):
    if not prompt:
        return None, "Please enter a prompt"

    width = int(width // 32) * 32
    height = int(height // 32) * 32
    num_frames = max(9, int((num_frames - 1) // 8) * 8 + 1)

    try:
        inf = get_inference()
        video_url, error = inf.generate_video(
            prompt=prompt, negative_prompt=negative_prompt,
            width=width, height=height, num_frames=num_frames,
            fps=fps, steps=steps, cfg_scale=cfg_scale,
            seed=int(seed) if seed else 42, input_image=input_image,
        )
        if error:
            return None, f"Failed: {error}"
        return video_url, "Video generated successfully!"
    except Exception as e:
        return None, f"Error: {str(e)}"

with gr.Blocks(title="Sulphur-2-base Video Generation", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""# Sulphur-2-base Video Generator\nBased on LTX Video 2.3 architecture. Supports T2V and I2V.""")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Input")
            prompt_input = gr.Textbox(label="Prompt", placeholder="Describe the video you want to generate...", lines=3)
            negative_prompt_input = gr.Textbox(label="Negative Prompt", lines=2, value="worst quality, blurry, distorted, jittery, inconsistent motion")
            input_image = gr.Image(label="Input Image (optional, for I2V)", type="filepath")

            with gr.Accordion("Advanced Settings", open=False):
                with gr.Row():
                    width_slider = gr.Slider(256, 1024, 768, step=32, label="Width")
                    height_slider = gr.Slider(256, 1024, 512, step=32, label="Height")
                with gr.Row():
                    frames_slider = gr.Slider(9, 257, 65, step=8, label="Frames (8n+1)")
                    fps_slider = gr.Slider(1, 60, 24, step=1, label="FPS")
                with gr.Row():
                    steps_slider = gr.Slider(1, 50, 20, step=1, label="Sampling Steps")
                    cfg_slider = gr.Slider(1.0, 20.0, 7.0, step=0.5, label="CFG Scale")
                seed_input = gr.Number(label="Seed", value=42, precision=0)

            generate_btn = gr.Button("Generate Video", variant="primary", size="lg")

        with gr.Column(scale=1):
            gr.Markdown("### Output")
            output_video = gr.Video(label="Generated Video")
            status_text = gr.Textbox(label="Status", interactive=False)

    generate_btn.click(
        fn=generate_video_fn,
        inputs=[prompt_input, negative_prompt_input, width_slider, height_slider,
                frames_slider, fps_slider, steps_slider, cfg_slider, seed_input, input_image],
        outputs=[output_video, status_text],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
