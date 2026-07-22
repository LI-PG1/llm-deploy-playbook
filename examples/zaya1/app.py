import gradio as gr
import os
import torch
import time
import sys
import signal
import atexit
import traceback

MODEL_PATH = os.environ.get("MODEL_PATH", "/gemini/pretrain")

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_SSL_VERIFY"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import os as _os
_os.environ.setdefault("LD_LIBRARY_PATH", "")
if "/opt/orion" not in _os.environ.get("LD_LIBRARY_PATH", ""):
    _os.environ["LD_LIBRARY_PATH"] = "/opt/orion/orion_runtime/gpu/cuda/orion-cuda-12.2/:" + _os.environ.get("LD_LIBRARY_PATH", "")

LOG = open("/tmp/debug.log", "w", buffering=1)

def dlog(msg):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    LOG.write(line + "\n")
    LOG.flush()

# Crash signal handlers
def _on_signal(sig, frame):
    dlog(f"SIGNAL:{sig}({signal.Signals(sig).name}) at {traceback.extract_stack(frame)[-1] if frame else '?'}")
    if sig == signal.SIGTERM:
        sys.exit(0)
    else:
        sys.exit(1)

for sig in [signal.SIGSEGV, signal.SIGABRT, signal.SIGTERM, signal.SIGINT]:
    try:
        signal.signal(sig, _on_signal)
    except Exception:
        pass

atexit.register(lambda: dlog("EXIT"))

# GPU memory snapshot
def gpu_info():
    try:
        a = torch.cuda.memory_allocated(0) / 1e9
        r = torch.cuda.memory_reserved(0) / 1e9
        return f"GPU alloc={a:.1f}GB reserved={r:.1f}GB"
    except Exception:
        return "GPU:N/A"

dlog("START")
dlog(gpu_info())

def load_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dlog("load_tokenizer_start")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    dlog("load_tokenizer_done")

    dlog(f"load_model_start {gpu_info()}")
    t0 = time.time()
    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            dtype=torch.float16,
            trust_remote_code=True,
            device_map="auto",
            local_files_only=True,
        )
        dlog(f"load_model_done {time.time()-t0:.0f}s {model.device} {gpu_info()}")
        return model, tokenizer
    except Exception as e:
        dlog(f"load_model_ERROR: {e}\n{traceback.format_exc()}")
        raise

dlog("INIT_load_start")
try:
    model, tokenizer = load_model()
    dlog("INIT_load_done")
except Exception as e:
    dlog(f"INIT_FATAL: {e}")
    sys.exit(1)

dlog("INIT_gradio_start")
dlog(gpu_info())

def chat_fn(message, history):
    if not message.strip():
        return history, history
    inputs = tokenizer(message, return_tensors="pt").to(model.device)
    input_len = inputs.input_ids.shape[-1]
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=512, do_sample=True, temperature=0.7, top_p=0.8, top_k=50, pad_token_id=tokenizer.eos_token_id)
    reply = tokenizer.decode(out[0][input_len:], skip_special_tokens=True)
    history.append((message, reply.strip() if reply.strip() else "(empty)"))
    return history, history

with gr.Blocks(title="ZAYA1-8B", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# ZAYA1-8B MoE")
    chatbot = gr.Chatbot(label="Chat", height=550, bubble_full_width=False)
    msg = gr.Textbox(label="Message", placeholder="Type here...", lines=2)
    send = gr.Button("Send", variant="primary")
    state = gr.State([])
    send.click(chat_fn, [msg, state], [chatbot, state]).then(lambda: "", None, msg)
    msg.submit(chat_fn, [msg, state], [chatbot, state]).then(lambda: "", None, msg)

if __name__ == "__main__":
    dlog("launching_gradio")
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
