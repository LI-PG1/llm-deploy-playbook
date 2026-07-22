# Sulphur-2-base 视频生成服务 Dockerfile
# 基础镜像: PyTorch 2.1.1 / CUDA 12.1 / Python 3.10 / Ubuntu 22.04

# === 系统依赖 ===
RUN sed -i 's/archive.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list
RUN apt-get update && apt-get install -y git wget unzip libgl1-mesa-glx && rm -rf /var/lib/apt/lists/*

# === Upgrade torch ===
RUN pip install --no-cache-dir --no-deps torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121
RUN pip install --no-cache-dir nvidia-cudnn-cu12==9.1.0.70 --index-url https://download.pytorch.org/whl/cu121
RUN pip install --no-cache-dir transformers==4.50.0

# === ComfyUI ===
RUN cd /opt && git clone --depth 1 https://gitee.com/mirrors/ComfyUI.git
RUN cd /opt/ComfyUI && pip install -r requirements.txt --index-url https://pypi.virtaicloud.com/repository/pypi/simple
RUN pip install av comfyui-frontend-package --index-url https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install comfy-kitchen --index-url https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install kornia simpleeval ninja modelscope --no-deps --index-url https://pypi.virtaicloud.com/repository/pypi/simple
RUN pip install --no-deps einops safetensors aiohttp yarl pyyaml Pillow scipy tqdm psutil alembic SQLAlchemy filelock requests --index-url https://pypi.virtaicloud.com/repository/pypi/simple

# === Environment ===
ENV LD_LIBRARY_PATH /root/miniconda3/lib:$LD_LIBRARY_PATH
ENV PYTHONPATH /opt/ComfyUI:$PYTHONPATH

# === Custom Nodes (files included via COPY) ===
COPY ComfyUI-LTXVideo-main.zip /tmp/
COPY ComfyUI-KJNodes.zip /tmp/
COPY ComfyUI-Math.zip /tmp/
RUN cd /opt/ComfyUI/custom_nodes && \
    unzip -o /tmp/ComfyUI-LTXVideo-main.zip && mv ComfyUI-LTXVideo-main ComfyUI-LTXVideo && \
    unzip -o /tmp/ComfyUI-KJNodes.zip && mv ComfyUI-KJNodes-main ComfyUI-KJNodes && \
    unzip -o /tmp/ComfyUI-Math.zip && mv ComfyMath-main ComfyMath && \
    rm /tmp/ComfyUI-*.zip

# === Tokenizer Patch ===
RUN python -c "
path='/opt/ComfyUI/comfy/text_encoders/lt.py'
with open(path) as f: c=f.read()
c=c.replace('tokenizer = tokenizer_data.get(\"spiece_model\", None)',
    'tokenizer = tokenizer_data.get(\"spiece_model\", None)\n            if tokenizer is None:\n                tokenizer = \"/opt/ComfyUI/models/text_encoders/gemma3_12b/tokenizer.model\"')
with open(path,'w') as f: f.write(c)
"

# === ComfyMathExpression Patch ===
RUN echo '
class ComfyMathExpression:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "values.a": ("FLOAT,INT", {"default": 0, "forceInput": True}),
            "values.b": ("FLOAT,INT", {"default": 0, "forceInput": False}),
            "expression": ("STRING", {"default": "a/2", "multiline": False})
        }}
    RETURN_TYPES = ("FLOAT", "INT",)
    FUNCTION = "eval_expr"
    CATEGORY = "math"
    def eval_expr(self, **kwargs):
        import eval
        a = kwargs.get("values.a", 0)
        b = kwargs.get("values.b", None)
        expr = kwargs.get("expression", "a")
        result = eval(expr, {"__builtins__": {}}, {"a": a, "b": b})
        return (float(result), int(result),)
' >> /opt/ComfyUI/custom_nodes/ComfyMath/__init__.py
