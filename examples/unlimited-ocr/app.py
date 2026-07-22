from transformers import AutoModel, AutoTokenizer
import gradio as gr
tokenizer = AutoTokenizer.from_pretrained("baidu/Unlimited-OCR", trust_remote_code=True)
model = AutoModel.from_pretrained("baidu/Unlimited-OCR", trust_remote_code=True, device_map="auto")
def ocr(image):
    result = model.infer(tokenizer, prompt="document parsing.", image_file=image)
    return result
gr.Interface(fn=ocr, inputs=gr.Image(type="pil"), outputs="text").launch()