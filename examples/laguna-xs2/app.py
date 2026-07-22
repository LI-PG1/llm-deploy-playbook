import os, torch, gradio as gr
from threading import Thread, Event
from urllib.request import urlopen

# benchmark_llm.py 需要的外部接口
MODEL_PATH = os.environ.get('MODEL_PATH', '/gemini/pretrain/Laguna-XS.2')
model = None
tokenizer = None
MODEL_LOADED = Event()

def load_model():
    global model, tokenizer
    if model is not None:
        return model, tokenizer
    print('Loading model...', flush=True)
    from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
    config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH, trust_remote_code=True, fix_mistral_regex=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, config=config, torch_dtype=torch.bfloat16,
        device_map='auto', trust_remote_code=True, local_files_only=True,
    )
    model.eval()
    MODEL_LOADED.set()
    print('Model ready!', flush=True)
    return model, tokenizer

# 后台加载模型，Gradio 先启动
Thread(target=load_model, daemon=True).start()

# benchmark_llm.py 兼容接口
def generate_response(messages, temperature=0.7, max_tokens=512, top_p=0.8):
    """与 benchmark_llm.py 兼容的接口"""
    load_model()
    MODEL_LOADED.wait()
    
    encoded = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors='pt',
        enable_thinking=True,
    )
    input_ids = encoded['input_ids'].to('cuda:0')
    attention_mask = encoded['attention_mask'].to('cuda:0')
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=(temperature > 0),
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=model.config.eos_token_id,
        )
    reply = tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)
    return reply

def respond(message, history, temperature, max_tokens):
    MODEL_LOADED.wait()  # 等模型加载完
    history_openai = []
    for user_msg, bot_msg in history:
        history_openai.append({"role": "user", "content": user_msg})
        if bot_msg:
            history_openai.append({"role": "assistant", "content": bot_msg})
    history_openai.append({"role": "user", "content": message})

    encoded = tokenizer.apply_chat_template(
        history_openai, add_generation_prompt=True, return_tensors='pt',
        enable_thinking=True,
    )
    input_ids = encoded['input_ids'].to('cuda:0')
    attention_mask = encoded['attention_mask'].to('cuda:0')

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=(temperature > 0),
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=model.config.eos_token_id,
        )
    reply = tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)
    return reply

with gr.Blocks(title='Laguna-XS.2 代码生成') as demo:
    gr.Markdown('# 【Poolside/Laguna-XS.2】代码生成推理服务')
    gr.Markdown('33B MoE（3B激活）代码生成模型，支持滑动窗口注意力与262K超长上下文。具备原生推理思考模式（thinking）与工具调用能力，专为编程问答、代码审查、Bug修复场景优化。')
    chatbot = gr.Chatbot(height=500)
    msg = gr.Textbox(label='输入编程问题')
    with gr.Accordion('参数', open=False):
        temp = gr.Slider(0.0, 1.5, 0.7, step=0.05, label='Temperature')
        max_tok = gr.Slider(128, 8192, 2048, step=128, label='最大生成长度')

    def chat(msg, hist, temp, max_tok):
        if not MODEL_LOADED.is_set():
            return '', hist + [('系统提示', '模型正在加载中（约需 2 小时），请稍后再试...')]
        reply = respond(msg, hist, temp, max_tok)
        hist.append((msg, reply))
        return '', hist

    msg.submit(chat, [msg, chatbot, temp, max_tok], [msg, chatbot])

demo.queue().launch(server_name='0.0.0.0', server_port=7860)
