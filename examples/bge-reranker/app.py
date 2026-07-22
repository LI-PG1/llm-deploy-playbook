import gradio as gr
import os
import torch
import time
from threading import Lock

MODEL_PATH = os.environ.get("MODEL_PATH", "/gemini/pretrain/bge-reranker-v2-m3")

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_SSL_VERIFY"] = "1"

MODEL = None
TOKENIZER = None
MODEL_LOCK = Lock()

def load_model():
    global MODEL, TOKENIZER
    if MODEL is not None:
        return MODEL, TOKENIZER

    with MODEL_LOCK:
        if MODEL is not None:
            return MODEL, TOKENIZER

        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        print(f"[LOAD] Loading tokenizer from {MODEL_PATH}...")
        TOKENIZER = AutoTokenizer.from_pretrained(
            MODEL_PATH,
            local_files_only=True,
        )

        print(f"[LOAD] Loading model from {MODEL_PATH}...")
        t0 = time.time()
        MODEL = AutoModelForSequenceClassification.from_pretrained(
            MODEL_PATH,
            local_files_only=True,
        )
        MODEL = MODEL.cuda()
        MODEL.eval()
        elapsed = time.time() - t0
        print(f"[LOAD] Model loaded in {elapsed:.1f}s")
        print(f"[LOAD] Device: {next(MODEL.parameters()).device}")

        return MODEL, TOKENIZER

def rerank(query, documents, top_k=None):
    model, tokenizer = load_model()

    if not query.strip():
        return "Error: Query cannot be empty."

    docs = [d.strip() for d in documents if d.strip()]
    if not docs:
        return "Error: No valid documents provided."

    pairs = [[query, doc] for doc in docs]

    print(f"[RERANK] Processing {len(docs)} documents...")
    t0 = time.time()

    with torch.no_grad():
        inputs = tokenizer(
            pairs,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(model.device)

        scores = model(**inputs, return_dict=True).logits.view(-1).float()
        scores = scores.cpu().tolist()

    elapsed = time.time() - t0
    print(f"[RERANK] Done in {elapsed:.1f}s ({len(docs)/elapsed:.1f} docs/s)")

    results = list(zip(docs, scores))
    results.sort(key=lambda x: x[1], reverse=True)

    if top_k and top_k > 0:
        results = results[:top_k]

    lines = []
    for rank, (doc, score) in enumerate(results, 1):
        lines.append(f"[Rank {rank}] Score: {score:.4f}\n{doc}\n")

    return "\n".join(lines)

def get_model_info():
    try:
        model, tokenizer = load_model()
        device = next(model.parameters()).device
        params = sum(p.numel() for p in model.parameters()) / 1e6
        return f"Model loaded on {device}\nParameters: {params:.1f}M\nTokenizer vocab size: {tokenizer.vocab_size}"
    except Exception as e:
        return f"Model not loaded yet: {e}"

with gr.Blocks(title="BGE Reranker v2-m3", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # BGE Reranker v2-m3

    BAAI Reranker: Cross-Encoder reranking model for improving search/retrieval results.
    Based on XLM-RoBERTa, supports 100+ languages.

    **Usage**: Enter a query and a list of candidate documents (one per line), click **Rerank** to get relevance-sorted results.

    Model: [BAAI/bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3)
    """)

    with gr.Row():
        with gr.Column(scale=1):
            with gr.Accordion("Model Info", open=True):
                model_info = gr.Textbox(
                    label="Status",
                    value="Model will load on first rerank request...",
                    lines=4,
                    interactive=False,
                )
                load_btn = gr.Button("Load Model", size="sm")

            with gr.Accordion("Examples", open=False):
                gr.Examples(
                    examples=[
                        [
                            "What is deep learning?",
                            "Deep learning is a subset of machine learning that uses neural networks with many layers.\nMachine learning is a field of artificial intelligence.\nThe weather today is sunny and warm.\nNeural networks are inspired by the human brain structure.\nPython is a popular programming language.",
                            5
                        ],
                        [
                            "Best practices for Docker containers",
                            "Docker containers should be lightweight and single-purpose.\nUse multi-stage builds to reduce image size.\nKubernetes is a container orchestration platform.\nAlways run containers as non-root users.\nLinux is an open-source operating system.",
                            3
                        ],
                    ],
                    inputs=[
                        gr.Textbox(label="Query", lines=2),
                        gr.Textbox(label="Documents (one per line)", lines=8),
                        gr.Number(label="Top-K"),
                    ],
                )

        with gr.Column(scale=2):
            query_input = gr.Textbox(
                label="Query",
                placeholder="Enter your search query here...",
                lines=2,
            )
            documents_input = gr.Textbox(
                label="Documents (one per line)",
                placeholder="Document 1: relevant content...\nDocument 2: another candidate...\nDocument 3: ...",
                lines=12,
            )
            with gr.Row():
                top_k_input = gr.Number(
                    label="Top-K (0 = show all)",
                    value=0,
                    precision=0,
                    minimum=0,
                )
                rerank_btn = gr.Button("Rerank", variant="primary", size="lg")

            results_output = gr.Textbox(
                label="Results (sorted by relevance score)",
                lines=15,
                interactive=False,
            )

    load_btn.click(
        fn=get_model_info,
        inputs=[],
        outputs=[model_info],
    )

    rerank_btn.click(
        fn=lambda q, docs, k: rerank(q, docs.split("\n"), int(k)),
        inputs=[query_input, documents_input, top_k_input],
        outputs=[results_output],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
