import gradio as gr
import os
import torch
import torch.nn.functional as F
import time
import numpy as np
from threading import Lock

MODEL_PATH = os.environ.get("MODEL_PATH", "/gemini/pretrain/gte-Qwen2-1.5B-instruct")

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_SSL_VERIFY"] = "1"

import os as _os
_os.environ.setdefault("LD_LIBRARY_PATH", "")
if "/opt/orion" not in _os.environ.get("LD_LIBRARY_PATH", ""):
    _os.environ["LD_LIBRARY_PATH"] = "/opt/orion/orion_runtime/gpu/cuda/orion-cuda-12.2/:" + _os.environ.get("LD_LIBRARY_PATH", "")

MODEL = None
TOKENIZER = None
MODEL_LOCK = Lock()

def last_token_pool(last_hidden_states, attention_mask):
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery: {query}'

def load_model():
    global MODEL, TOKENIZER
    if MODEL is not None:
        return MODEL, TOKENIZER

    with MODEL_LOCK:
        if MODEL is not None:
            return MODEL, TOKENIZER

        from transformers import AutoTokenizer, AutoModel

        print(f"[LOAD] Loading tokenizer from {MODEL_PATH}...")
        TOKENIZER = AutoTokenizer.from_pretrained(
            MODEL_PATH,
            local_files_only=True,
        )

        print(f"[LOAD] Loading model from {MODEL_PATH}...")
        t0 = time.time()
        MODEL = AutoModel.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            local_files_only=True,
        )
        MODEL = MODEL.cuda()
        MODEL.eval()
        elapsed = time.time() - t0
        print(f"[LOAD] Model loaded in {elapsed:.1f}s on cuda")

        return MODEL, TOKENIZER

def encode_texts(texts, task_description=None, max_length=8192):
    model, tokenizer = load_model()

    if task_description:
        texts = [get_detailed_instruct(task_description, t) for t in texts]

    batch_dict = tokenizer(
        texts,
        max_length=max_length,
        padding=True,
        truncation=True,
        return_tensors='pt',
    )
    batch_dict = {k: v.to(model.device) for k, v in batch_dict.items()}

    with torch.no_grad():
        outputs = model(**batch_dict)
        embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
        embeddings = F.normalize(embeddings, p=2, dim=1)

    return embeddings.cpu().float().numpy()

def embed_single(text, task_description):
    if not text.strip():
        return "Please enter text to encode.", ""

    t0 = time.time()
    try:
        embedding = encode_texts([text.strip()], task_description=task_description.strip() or None)
        elapsed = time.time() - t0

        vec = embedding[0]
        info = f"Embedding dimension: {len(vec)}\n"
        info += f"Norm: {np.linalg.norm(vec):.6f}\n"
        info += f"Processing time: {elapsed:.2f}s\n"
        info += f"\nFirst 20 values:\n{vec[:20]}\n"
        info += f"\nLast 20 values:\n{vec[-20:]}"

        dims_show = min(len(vec), 200)
        vec_preview = ", ".join([f"{v:.6f}" for v in vec[:dims_show]])
        if len(vec) > dims_show:
            vec_preview += f", ... (+{len(vec) - dims_show} more)"

        return info, vec_preview
    except Exception as e:
        return f"Error: {e}", ""

def compute_similarity(text1, text2, query_task):
    if not text1.strip() or not text2.strip():
        return "Please enter both texts."

    t0 = time.time()
    try:
        embeddings = encode_texts([text1.strip(), text2.strip()], task_description=query_task.strip() or None)
        sim = float(np.dot(embeddings[0], embeddings[1]))
        elapsed = time.time() - t0

        percentage = sim * 100
        if percentage >= 80:
            level = "Very High"
        elif percentage >= 60:
            level = "High"
        elif percentage >= 40:
            level = "Moderate"
        elif percentage >= 20:
            level = "Low"
        else:
            level = "Very Low"

        return f"Cosine Similarity: {sim:.6f} ({percentage:.1f}%)\nSimilarity Level: {level}\nTime: {elapsed:.2f}s"
    except Exception as e:
        return f"Error: {e}"

def search_documents(query, documents_text, task_description, top_k):
    if not query.strip():
        return "Please enter a query."
    if not documents_text.strip():
        return "Please enter at least one document."

    documents = [d.strip() for d in documents_text.strip().split("\n") if d.strip()]
    if not documents:
        return "No valid documents found."

    t0 = time.time()
    try:
        query_embeddings = encode_texts([query.strip()], task_description=task_description.strip() or None)
        doc_embeddings = encode_texts(documents)

        scores = (query_embeddings @ doc_embeddings.T).flatten()
        ranked = sorted(zip(range(len(documents)), scores, documents), key=lambda x: x[1], reverse=True)

        top_k = min(int(top_k), len(ranked))
        elapsed = time.time() - t0

        result = f"Results (top {top_k} of {len(documents)}, {elapsed:.2f}s):\n\n"
        for rank, (idx, score, doc) in enumerate(ranked[:top_k], 1):
            percentage = float(score) * 100
            result += f"#{rank} [Score: {percentage:.1f}%]\n"
            preview = doc[:200] + "..." if len(doc) > 200 else doc
            result += f"  {preview}\n\n"

        return result
    except Exception as e:
        return f"Error: {e}"

with gr.Blocks(title="gte-Qwen2-1.5B Embedding", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # gte-Qwen2-1.5B-Instruct — Text Embedding & Similarity

    **Alibaba NLP gte-Qwen2-1.5B-instruct**: 1.5B multilingual text embedding model.
    Built on Qwen2-1.5B with bidirectional attention & instruction tuning.
    32K max tokens, 1536-dim embeddings.

    Model: [Alibaba-NLP/gte-Qwen2-1.5B-instruct](https://huggingface.co/Alibaba-NLP/gte-Qwen2-1.5B-instruct)
    """)

    with gr.Tabs():
        with gr.TabItem("Text Embedding"):
            with gr.Row():
                with gr.Column(scale=2):
                    embed_input = gr.Textbox(
                        label="Input Text",
                        placeholder="Enter text to encode into a vector...",
                        lines=4,
                    )
                    embed_task = gr.Textbox(
                        label="Task Description (optional, for query-side instruction tuning)",
                        placeholder="e.g. Given a web search query, retrieve relevant passages that answer the query",
                        lines=2,
                    )
                    embed_btn = gr.Button("Generate Embedding", variant="primary")

                with gr.Column(scale=3):
                    embed_info = gr.Textbox(
                        label="Embedding Info",
                        lines=10,
                    )
                    embed_vector = gr.Textbox(
                        label="Embedding Vector",
                        lines=6,
                    )

            embed_btn.click(
                fn=embed_single,
                inputs=[embed_input, embed_task],
                outputs=[embed_info, embed_vector],
            )

        with gr.TabItem("Text Similarity"):
            with gr.Row():
                with gr.Column():
                    sim_text1 = gr.Textbox(
                        label="Text 1",
                        placeholder="Enter first text...",
                        lines=3,
                    )
                    sim_text2 = gr.Textbox(
                        label="Text 2",
                        placeholder="Enter second text...",
                        lines=3,
                    )
                    sim_task = gr.Textbox(
                        label="Task Description (optional)",
                        placeholder="Optional query-side instruction...",
                        lines=2,
                    )
                    sim_btn = gr.Button("Compute Similarity", variant="primary")

                with gr.Column():
                    sim_output = gr.Textbox(
                        label="Similarity Result",
                        lines=6,
                    )

            sim_btn.click(
                fn=compute_similarity,
                inputs=[sim_text1, sim_text2, sim_task],
                outputs=[sim_output],
            )

        with gr.TabItem("Document Search"):
            with gr.Row():
                with gr.Column(scale=1):
                    search_query = gr.Textbox(
                        label="Search Query",
                        placeholder="Enter your search query...",
                        lines=2,
                    )
                    search_task = gr.Textbox(
                        label="Task Description (optional)",
                        placeholder="e.g. Given a web search query, retrieve relevant passages that answer the query",
                        lines=2,
                    )
                    search_topk = gr.Slider(
                        1, 20, 5, step=1,
                        label="Top-K Results",
                    )
                    search_btn = gr.Button("Search", variant="primary")

                with gr.Column(scale=2):
                    search_docs = gr.Textbox(
                        label="Documents (one per line)",
                        placeholder="Paste documents here, one per line...\n\nExample:\nAs a general guideline, the CDC recommends 46 grams of protein per day for women.\nProtein is an essential macronutrient for muscle growth and repair.",
                        lines=12,
                    )
                    search_output = gr.Textbox(
                        label="Search Results",
                        lines=12,
                    )

            search_btn.click(
                fn=search_documents,
                inputs=[search_query, search_docs, search_task, search_topk],
                outputs=[search_output],
            )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
