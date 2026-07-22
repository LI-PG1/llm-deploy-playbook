import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import huggingface_hub
if not hasattr(huggingface_hub, "HfFolder"):
    class _FakeHfFolder:
        @staticmethod
        def get_token():
            return None
        @staticmethod
        def login(token=None, **kwargs):
            pass
    huggingface_hub.HfFolder = _FakeHfFolder

import gradio as gr
from supertonic import TTS
import argparse
import time
import traceback

LANGUAGES = {
    "Arabic (ar)": "ar",
    "Bulgarian (bg)": "bg",
    "Croatian (hr)": "hr",
    "Czech (cs)": "cs",
    "Danish (da)": "da",
    "Dutch (nl)": "nl",
    "English (en)": "en",
    "Estonian (et)": "et",
    "Finnish (fi)": "fi",
    "French (fr)": "fr",
    "German (de)": "de",
    "Greek (el)": "el",
    "Hindi (hi)": "hi",
    "Hungarian (hu)": "hu",
    "Indonesian (id)": "id",
    "Italian (it)": "it",
    "Japanese (ja)": "ja",
    "Korean (ko)": "ko",
    "Latvian (lv)": "lv",
    "Lithuanian (lt)": "lt",
    "Polish (pl)": "pl",
    "Portuguese (pt)": "pt",
    "Romanian (ro)": "ro",
    "Russian (ru)": "ru",
    "Slovak (sk)": "sk",
    "Slovenian (sl)": "sl",
    "Spanish (es)": "es",
    "Swedish (sv)": "sv",
    "Turkish (tr)": "tr",
    "Ukrainian (uk)": "uk",
    "Vietnamese (vi)": "vi",
}

EXPRESSIONS = ["None", "laugh", "breath", "sigh"]

OUTPUT_DIR = "/tmp/output"

tts = None

def get_tts():
    global tts
    if tts is None:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        tts = TTS(auto_download=True)
    return tts

def get_available_voices():
    try:
        t = get_tts()
        voices = []
        for name in ["M1", "F1", "M2", "F2", "M3", "F3"]:
            try:
                t.get_voice_style(voice_name=name)
                voices.append(name)
            except Exception:
                pass
        if not voices:
            voices = ["M1", "F1"]
        return voices
    except Exception:
        return ["M1", "F1"]

def synthesize_speech(text, language_display, voice_name, expression):
    if not text or not text.strip():
        return None, "Please enter text"

    lang_code = LANGUAGES.get(language_display, "en")

    if expression and expression != "None":
        text = f"<{expression}>{text}"

    try:
        t = get_tts()
        style = t.get_voice_style(voice_name=voice_name)
        wav, duration = t.synthesize(text, voice_style=style, lang=lang_code)

        filename = f"supertone3_{int(time.time())}.wav"
        output_path = os.path.join(OUTPUT_DIR, filename)
        t.save_audio(wav, output_path)

        return output_path, f"Generated ({float(duration):.1f}s)"
    except Exception as e:
        traceback.print_exc()
        return None, f"Error: {str(e)}"

with gr.Blocks(title="Supertone-3 TTS", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # Supertone-3 Text-to-Speech

    31-language on-device TTS powered by ONNX Runtime. CPU inference, no GPU needed.
    Model: [Supertone/supertonic-3](https://hf-mirror.com/Supertone/supertonic-3)
    """)

    with gr.Row():
        with gr.Column(scale=2):
            text_input = gr.Textbox(
                label="Text",
                placeholder="Enter text to synthesize...",
                lines=5,
            )

            with gr.Row():
                lang_dropdown = gr.Dropdown(
                    choices=list(LANGUAGES.keys()),
                    value="English (en)",
                    label="Language",
                )
                voice_dropdown = gr.Dropdown(
                    choices=get_available_voices(),
                    value="M1",
                    label="Voice Style",
                )

            expression_dropdown = gr.Dropdown(
                choices=EXPRESSIONS,
                value="None",
                label="Expression Tag",
            )

            generate_btn = gr.Button("Generate Speech", variant="primary", size="lg")

        with gr.Column(scale=1):
            audio_output = gr.Audio(label="Output", type="filepath")
            status_text = gr.Textbox(label="Status", interactive=False)

    generate_btn.click(
        fn=synthesize_speech,
        inputs=[text_input, lang_dropdown, voice_dropdown, expression_dropdown],
        outputs=[audio_output, status_text],
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--server_name", type=str, default="0.0.0.0")
    parser.add_argument("--server_port", type=int, default=7860)
    args = parser.parse_args()
    demo.launch(server_name=args.server_name, server_port=args.server_port, share=False)
