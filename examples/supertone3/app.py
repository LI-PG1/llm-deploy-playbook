from supertonic import TTS
import gradio as gr
tts = TTS(auto_download=True)
def speak(text):
    style = tts.get_voice_style("M1")
    wav, _ = tts.synthesize(text, voice_style=style, lang="zh")
    return ("output.wav", wav)
gr.Interface(fn=speak, inputs="text", outputs="audio").launch()