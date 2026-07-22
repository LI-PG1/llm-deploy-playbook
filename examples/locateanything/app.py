from transformers import AutoProcessor, AutoModelForCausalLM
import gradio as gr
processor = AutoProcessor.from_pretrained("nvidia/LocateAnything-3B", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("nvidia/LocateAnything-3B", trust_remote_code=True, device_map="auto")
def detect(image, prompt):
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=8192)
    return processor.decode(outputs[0], skip_special_tokens=True)
gr.Interface(fn=detect, inputs=[gr.Image(type="pil"), "text"], outputs="text").launch()