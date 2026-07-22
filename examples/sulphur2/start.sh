#!/bin/bash

echo "============================================"
echo "  Sulphur-2-base Video Generation Service"
echo "============================================"

# === 环境变量 ===
export LD_LIBRARY_PATH=/root/miniconda3/lib:$LD_LIBRARY_PATH
export PYTHONPATH=/opt/ComfyUI:$PYTHONPATH
export COMFYUI_PATH=/opt/ComfyUI
export SULPHUR2_MODEL_PATH=/gemini/pretrain/

mkdir -p /tmp/output
mkdir -p /opt/ComfyUI/custom_nodes

# === 配置模型路径 ===
echo ""
echo "[1/4] Configuring model paths..."
cat > /opt/ComfyUI/extra_model_paths.yaml << 'YAMLEOF'
comfyui:
    base_path: /opt/ComfyUI/
    checkpoints: /gemini/pretrain/
    vae: /gemini/pretrain/
    clip: /gemini/pretrain/
YAMLEOF
echo "[OK] extra_model_paths.yaml configured"

# === GPU 直载补丁 ===
echo ""
echo "[2/5] Applying GPU direct-load patch..."
cp /gemini/code/gpu_load_patch.py /opt/ComfyUI/comfy/gpu_load_patch.py
sed -i 's/^from comfy.cli_args import args/from comfy.cli_args import args\nimport comfy.gpu_load_patch/' /opt/ComfyUI/main.py
echo "[OK] GPU patch applied"

# === 启动 ComfyUI 服务端 ===
echo ""
echo "[3/5] Starting ComfyUI server (lowvram mode)..."
cd /opt/ComfyUI
python main.py --listen 0.0.0.0 --port 8188 --disable-auto-launch --output-directory /tmp/output --lowvram --fast &
COMFY_PID=$!

# 等待 ComfyUI 就绪
echo -n "Waiting for ComfyUI to be ready"
for i in $(seq 1 30); do
    if curl -s http://127.0.0.1:8188/system_stats > /dev/null 2>&1; then
        echo ""
        echo "[OK] ComfyUI is ready (PID: $COMFY_PID)"
        break
    fi
    echo -n "."
    sleep 2
done
echo ""

# === 运行测试 ===
echo "[4/5] Running test API..."
python /gemini/code/test_api.py

# === 启动 Gradio Web UI ===
echo "[5/5] Starting Gradio Web UI..."
cd /gemini/code
python app.py &
GRADIO_PID=$!

echo ""
echo "============================================"
echo "  Service started!"
echo "  ComfyUI PID: $COMFY_PID (port 8188)"
echo "  Gradio PID:  $GRADIO_PID (port 7860)"
echo "============================================"

wait
