import gradio as gr
import os
import torch
import time
from threading import Lock

MODEL_PATH = os.environ.get("MODEL_PATH", "/gemini/pretrain/Hy-MT2-7B-model")

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_SSL_VERIFY"] = "1"

MODEL = None
TOKENIZER = None
MODEL_LOCK = Lock()

LANGUAGE_NAMES = [
    "中文", "英语", "日语", "韩语", "法语", "德语", "西班牙语",
    "葡萄牙语", "意大利语", "俄语", "阿拉伯语", "泰语", "越南语",
    "印尼语", "马来语", "菲律宾语", "印地语", "土耳其语", "波兰语",
    "捷克语", "荷兰语", "高棉语", "缅甸语", "波斯语", "古吉拉特语",
    "乌尔都语", "泰卢固语", "马拉地语", "希伯来语", "孟加拉语",
    "泰米尔语", "乌克兰语", "繁体中文", "藏语", "哈萨克语",
    "蒙古语", "维吾尔语", "粤语",
]

ENGLISH_NAMES = {
    "中文": "Chinese", "英语": "English", "日语": "Japanese",
    "韩语": "Korean", "法语": "French", "德语": "German",
    "西班牙语": "Spanish", "葡萄牙语": "Portuguese",
    "意大利语": "Italian", "俄语": "Russian", "阿拉伯语": "Arabic",
    "泰语": "Thai", "越南语": "Vietnamese", "印尼语": "Indonesian",
    "马来语": "Malay", "菲律宾语": "Filipino", "印地语": "Hindi",
    "土耳其语": "Turkish", "波兰语": "Polish", "捷克语": "Czech",
    "荷兰语": "Dutch", "高棉语": "Khmer", "缅甸语": "Burmese",
    "波斯语": "Persian", "古吉拉特语": "Gujarati", "乌尔都语": "Urdu",
    "泰卢固语": "Telugu", "马拉地语": "Marathi", "希伯来语": "Hebrew",
    "孟加拉语": "Bengali", "泰米尔语": "Tamil", "乌克兰语": "Ukrainian",
    "繁体中文": "Traditional Chinese", "藏语": "Tibetan",
    "哈萨克语": "Kazakh", "蒙古语": "Mongolian", "维吾尔语": "Uyghur",
    "粤语": "Cantonese",
}

def load_model():
    global MODEL, TOKENIZER
    if MODEL is not None:
        return MODEL, TOKENIZER

    with MODEL_LOCK:
        if MODEL is not None:
            return MODEL, TOKENIZER

        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"[LOAD] Loading tokenizer from {MODEL_PATH}...")
        TOKENIZER = AutoTokenizer.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True,
            local_files_only=True,
        )

        print(f"[LOAD] Loading model from {MODEL_PATH}...")
        t0 = time.time()
        MODEL = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            local_files_only=True,
        )
        MODEL.eval()
        elapsed = time.time() - t0
        print(f"[LOAD] Model loaded in {elapsed:.1f}s")
        print(f"[LOAD] Device: {MODEL.device}")

        return MODEL, TOKENIZER

def build_prompt(source_lang, target_lang, source_text):
    use_english = source_lang in ENGLISH_NAMES and target_lang in ENGLISH_NAMES
    if use_english:
        src = ENGLISH_NAMES[source_lang]
        tgt = ENGLISH_NAMES[target_lang]
        return f"Translate the following text into {tgt}. Note that you should only output the translated result without any additional explanation:\n\n{source_text}"
    else:
        return f"将以下文本翻译成{target_lang}，注意只需要输出翻译后的结果，不要额外解释：\n\n{source_text}"

def translate(source_lang, target_lang, source_text, temperature, top_p, top_k,
              repetition_penalty, max_tokens):
    if not source_text.strip():
        return "", "⚠️ 请输入要翻译的文本"

    try:
        model, tokenizer = load_model()

        prompt = build_prompt(source_lang, target_lang, source_text)
        messages = [{"role": "user", "content": prompt}]

        tokenized = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
        )
        tokenized = tokenized.to(model.device)
        input_len = tokenized["input_ids"].shape[-1]
        print(f"[INFER] Input tokens: {input_len}")

        t0 = time.time()
        with torch.no_grad():
            generated = model.generate(
                **tokenized,
                max_new_tokens=int(max_tokens),
                temperature=float(temperature),
                top_p=float(top_p),
                top_k=int(top_k),
                repetition_penalty=float(repetition_penalty),
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        elapsed = time.time() - t0

        output_tokens = generated[0][input_len:]
        response = tokenizer.decode(output_tokens, skip_special_tokens=True)

        print(f"[INFER] Output: {len(output_tokens)} tokens, {elapsed:.1f}s, {len(output_tokens) / elapsed:.1f} tok/s")

        return response.strip(), f"✅ 翻译完成 | {len(output_tokens)} tokens | {elapsed:.1f}s | {len(output_tokens) / elapsed:.1f} tok/s"

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return "ERROR", f"❌ {type(e).__name__}: {e}"

def swap_languages(source_lang, target_lang):
    return target_lang, source_lang

with gr.Blocks(title="Hy-MT2-7B Translator", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🌐 Hy-MT2-7B 多语言翻译

    **腾讯混元** 开源的多语言翻译模型，支持 **33种语言** 互译。
    WMT26 翻译竞赛官方合作模型。7B 参数，单卡即可部署。

    📄 [模型页面](https://huggingface.co/tencent/Hy-MT2-7B) | 📝 [技术报告](https://arxiv.org/pdf/2605.22064)
    """)

    with gr.Row():
        with gr.Column(scale=2):
            source_lang = gr.Dropdown(
                choices=LANGUAGE_NAMES,
                value="中文",
                label="源语言 (Source Language)",
            )
            source_text = gr.Textbox(
                label="输入文本",
                placeholder="请输入要翻译的文本...",
                lines=8,
            )

        with gr.Column(scale=1):
            with gr.Row():
                swap_btn = gr.Button("⇄ 交换语言", variant="secondary", size="sm")

        with gr.Column(scale=2):
            target_lang = gr.Dropdown(
                choices=LANGUAGE_NAMES,
                value="英语",
                label="目标语言 (Target Language)",
            )
            translated_text = gr.Textbox(
                label="翻译结果",
                placeholder="翻译结果将显示在这里...",
                lines=8,
                interactive=False,
            )

    with gr.Row():
        translate_btn = gr.Button("🌐 翻译", variant="primary", size="lg")

    status_text = gr.Textbox(label="", value="", interactive=False, visible=False)
    status_display = gr.Markdown("")

    with gr.Accordion("⚙️ 高级参数 (Advanced Parameters)", open=False):
        with gr.Row():
            temperature = gr.Slider(0.1, 2.0, 0.7, step=0.05, label="Temperature")
            top_p = gr.Slider(0.1, 1.0, 0.6, step=0.05, label="Top-P")
        with gr.Row():
            top_k = gr.Slider(1, 100, 20, step=1, label="Top-K")
            repetition_penalty = gr.Slider(1.0, 2.0, 1.05, step=0.01, label="Repetition Penalty")
        with gr.Row():
            max_tokens = gr.Slider(128, 4096, 4096, step=64, label="Max New Tokens")

    translate_btn.click(
        fn=translate,
        inputs=[source_lang, target_lang, source_text, temperature, top_p,
                top_k, repetition_penalty, max_tokens],
        outputs=[translated_text, status_text],
    ).then(
        fn=lambda s: gr.Markdown(s),
        inputs=[status_text],
        outputs=[status_display],
    )

    swap_btn.click(
        fn=swap_languages,
        inputs=[source_lang, target_lang],
        outputs=[source_lang, target_lang],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
