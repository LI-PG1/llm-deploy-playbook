#!/usr/bin/env python3
"""
Kronos 金融 K 线预测 WebUI
基于 NeoQuasar/Kronos 时序预测基础模型
"""

import os, sys, json, tempfile, traceback
from pathlib import Path

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_SSL_VERIFY"] = "1"

import gradio as gr
import pandas as pd
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# ─── 路径配置 ─────────────────────────────────
MODEL_BASE = "/gemini/pretrain/Kronos"
TOKENIZER_PATH = os.path.join(MODEL_BASE, "Kronos-Tokenizer-base")
SMALL_PATH = os.path.join(MODEL_BASE, "Kronos-small")
BASE_PATH = os.path.join(MODEL_BASE, "Kronos-base")

# 模型代码路径（启动时下载 Kronos 仓库，用 Gitee 镜像）
KRONOS_REPO = "https://gitee.com/persiacat/Kronos.git"
KRONOS_CODE_DIR = "/tmp/kronos_code"


def ensure_kronos_code():
    """克隆 Kronos GitHub 仓库获取模型代码"""
    if not os.path.exists(os.path.join(KRONOS_CODE_DIR, "model", "__init__.py")):
        import subprocess
        print(f"[SETUP] Cloning Kronos repo from {KRONOS_REPO}...")
        subprocess.run(
            ["git", "clone", "--depth", "1", KRONOS_REPO, KRONOS_CODE_DIR],
            capture_output=True, text=True,
        )
        print("[SETUP] Kronos code ready")
    sys.path.insert(0, KRONOS_CODE_DIR)
    print(f"[SETUP] Kronos code path added: {KRONOS_CODE_DIR}")


def _load_local_model(model_class, model_dir):
    """从本地目录加载 Kronos 模型/分词器，完全绕过 huggingface_hub"""
    import safetensors.torch
    config_path = os.path.join(model_dir, "config.json")
    weights_path = os.path.join(model_dir, "model.safetensors")
    with open(config_path) as f:
        config = json.load(f)
    instance = model_class(**config)
    weights = safetensors.torch.load_file(weights_path, device="cpu")
    instance.load_state_dict(weights)
    return instance


def load_models(model_size="small"):
    """加载分词器和模型（绕过 huggingface_hub）"""
    ensure_kronos_code()
    from model import Kronos, KronosTokenizer, KronosPredictor

    tokenizer_path = TOKENIZER_PATH
    model_path = SMALL_PATH if model_size == "small" else BASE_PATH

    print(f"[LOAD] Tokenizer: {tokenizer_path}")
    tokenizer = _load_local_model(KronosTokenizer, tokenizer_path)

    print(f"[LOAD] Model: {model_path} ({model_size})")
    model = _load_local_model(Kronos, model_path)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    print(f"[LOAD] Device: {device}")

    predictor = KronosPredictor(
        model, tokenizer,
        device=device,
        max_context=512,
    )
    return predictor


def parse_csv(file_obj):
    """解析上传的 CSV 文件"""
    df = pd.read_csv(file_obj.name if hasattr(file_obj, "name") else file_obj)
    # 自动识别时间列
    time_col = None
    for col in df.columns:
        if col.lower() in ("timestamp", "timestamps", "time", "date", "datetime", "trading_date"):
            time_col = col
            break

    # 检查必需的 OHLCV 列
    required = {"open", "high", "low", "close"}
    df_cols_lower = {c.lower(): c for c in df.columns}
    missing = required - set(df_cols_lower.keys())
    if missing:
        raise ValueError(f"CSV 缺少必需的列: {missing}，已有列: {list(df.columns)}")

    # 重命名为标准列名
    rename_map = {v: k for k, v in df_cols_lower.items() if k in required}
    df = df.rename(columns=rename_map)

    # 额外列
    extra_cols = ["open", "high", "low", "close"]
    if "volume" in df_cols_lower:
        extra_cols.append("volume")
    if "amount" in df_cols_lower:
        extra_cols.append("amount")

    x_df = df[extra_cols].copy()

    # 时间戳
    if time_col:
        timestamps = pd.to_datetime(df[time_col])
    else:
        timestamps = pd.date_range(
            end=pd.Timestamp.now(),
            periods=len(df),
            freq="5min",
        )

    return x_df, timestamps, df[[c for c in df.columns if c not in extra_cols + ([time_col] if time_col else [])] + extra_cols + ([time_col] if time_col else [])]


def generate_sample_data():
    """生成示例数据（模拟 BTC 价格）"""
    np.random.seed(42)
    n = 800
    base_price = 50000.0
    prices = [base_price]
    for i in range(1, n):
        ret = np.random.normal(0, 0.002)
        prices.append(prices[-1] * (1 + ret))

    df = pd.DataFrame({
        "timestamp": pd.date_range(end=pd.Timestamp.now(), periods=n, freq="5min"),
        "open": prices,
        "high": [p * (1 + abs(np.random.normal(0, 0.001))) for p in prices],
        "low":  [p * (1 - abs(np.random.normal(0, 0.001))) for p in prices],
        "close": [p * (1 + np.random.normal(0, 0.001)) for p in prices],
        "volume": np.random.uniform(100, 1000, n),
        "amount": np.random.uniform(5000000, 50000000, n),
    })
    return df


def run_prediction(
    model_size, csv_file, use_sample,
    lookback, pred_len, temperature, top_p, sample_count,
):
    """执行预测并返回结果图表"""
    try:
        # 1. 准备数据
        if use_sample or csv_file is None:
            df = generate_sample_data()
            x_df = df[["open", "high", "low", "close", "volume", "amount"]].iloc[:lookback]
            x_ts = df["timestamp"].iloc[:lookback]
            y_ts = df["timestamp"].iloc[lookback:lookback+pred_len]
            source_info = "Sample Data (Simulated BTC/USDT 5min K-line)"
            full_df = df
        else:
            x_df, timestamps, full_df = parse_csv(csv_file)
            x_df = x_df.iloc[:lookback]
            x_ts = timestamps.iloc[:lookback]
            # 补全未来时间戳（CSV 可能不够长）
            if len(timestamps) < lookback + pred_len:
                last_ts = timestamps.iloc[-1]
                freq = (timestamps.iloc[-1] - timestamps.iloc[-2]) if len(timestamps) >= 2 else pd.Timedelta(minutes=5)
                extra = pd.date_range(start=last_ts + freq, periods=lookback + pred_len - len(timestamps), freq=freq)
                y_ts = pd.concat([timestamps.iloc[lookback:], pd.Series(extra)])
            else:
                y_ts = timestamps.iloc[lookback:lookback+pred_len]
            source_info = f"Uploaded: {os.path.basename(csv_file.name)}"

        # 2. 加载模型
        predictor = load_models(model_size)

        # 3. 运行预测
        print(f"[PREDICT] lookback={lookback}, pred_len={pred_len}, "
              f"T={temperature}, top_p={top_p}, sample_count={sample_count}")
        pred_df = predictor.predict(
            df=x_df,
            x_timestamp=x_ts,
            y_timestamp=y_ts,
            pred_len=pred_len,
            T=temperature,
            top_p=top_p,
            sample_count=sample_count,
        )

        # 4. 绘制结果
        fig, axes = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={"height_ratios": [3, 1]})
        fig.suptitle(f"Kronos {model_size.upper()} - K-line Price Prediction", fontsize=14, y=0.98)

        # 主图：价格预测
        ax1 = axes[0]
        # 历史数据
        hist_close = x_df["close"].values
        hist_idx = np.arange(len(hist_close))
        ax1.plot(hist_idx, hist_close, "b-", linewidth=1.5, alpha=0.7, label="Historical Close")

        # 预测数据
        pred_start = len(hist_close)
        pred_idx = np.arange(pred_start, pred_start + len(pred_df))

        if "close" in pred_df.columns:
            ax1.plot(pred_idx, pred_df["close"].values, "r--", linewidth=2,
                     marker="o", markersize=3, label="Predicted Close")
            # 置信区间（如果有）
            for col in pred_df.columns:
                if "upper" in col.lower():
                    ax1.fill_between(pred_idx, pred_df["close"].values,
                                     pred_df[col].values, alpha=0.15, color="red")

        # 真实值（示例数据有真实值才显示）
        if not use_sample and len(full_df) > lookback + pred_len:
            actual = full_df["close"].values[lookback:lookback+pred_len]
            if len(actual) == len(pred_df):
                ax1.plot(pred_idx, actual, "g-", linewidth=1.5, alpha=0.6, label="Actual")

        ax1.axvline(x=pred_start - 0.5, color="gray", linestyle=":", alpha=0.5)
        ax1.text(pred_start - 0.5, ax1.get_ylim()[1] * 0.95, "← History | Prediction →",
                 ha="center", fontsize=9, color="gray", alpha=0.7)
        ax1.set_ylabel("Price")
        ax1.legend(loc="best", fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.set_title(f"Source: {source_info}  |  Pred Length: {pred_len}  Samples: {sample_count}")

        # 副图：涨跌幅
        ax2 = axes[1]
        if "close" in pred_df.columns:
            pred_close = pred_df["close"].values
            last_hist = hist_close[-1]
            returns = (pred_close / last_hist - 1) * 100
            colors = ["g" if r >= 0 else "r" for r in returns]
            ax2.bar(pred_idx, returns, color=colors, alpha=0.7, width=0.6)
            ax2.axhline(y=0, color="black", linewidth=0.5)
            ax2.set_ylabel("Change (%)")
            ax2.set_xlabel("Time Step")
            ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        # 5. 构建结果表格
        result_html = pred_df.head(20).to_html(
            classes="table table-striped table-bordered",
            float_format=lambda x: f"{x:.4f}",
        )

        return fig, result_html, f"预测完成！共 {len(pred_df)} 个时间步"

    except Exception as e:
        traceback.print_exc()
        return None, "", f"错误: {type(e).__name__}: {str(e)}"


# ─── Gradio UI ─────────────────────────────────
with gr.Blocks(
    title="Kronos 金融K线预测",
    theme=gr.themes.Soft(),
    css="""
    .kronos-header { text-align: center; margin-bottom: 1em; }
    .kronos-header h1 { margin-bottom: 0.2em; }
    .kronos-header p { color: #666; font-size: 0.95em; }
    """,
) as demo:
    gr.HTML("""
    <div class="kronos-header">
        <h1>📈 Kronos 金融K线预测</h1>
        <p>基于 NeoQuasar/Kronos 时序预测基础模型 | 支持 BTC/股票/期货 K 线预测</p>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=1):
            # ─── 模型配置 ───
            with gr.Group():
                gr.Markdown("### 模型配置")
                model_selector = gr.Radio(
                    choices=[("Kronos-small (24.7M 参数, 快速)", "small"),
                             ("Kronos-base (102.3M 参数, 更准)", "base")],
                    value="small",
                    label="模型",
                )

            # ─── 数据输入 ───
            with gr.Group():
                gr.Markdown("### 数据输入")
                use_sample_cb = gr.Checkbox(
                    value=True,
                    label="使用示例数据（模拟 BTC/USDT）",
                )
                csv_upload = gr.File(
                    label="上传 CSV 文件（列: timestamp, open, high, low, close, volume, amount）",
                    file_types=[".csv"],
                    visible=False,
                )

                def toggle_csv(use_sample):
                    return gr.update(visible=not use_sample)
                use_sample_cb.change(toggle_csv, inputs=[use_sample_cb], outputs=[csv_upload])

                gr.Markdown("""
                **CSV 格式要求:**
                - 必需列: `timestamp`, `open`, `high`, `low`, `close`
                - 可选列: `volume`, `amount`
                - 时间列名支持: timestamp, time, date, datetime
                """)

            # ─── 预测参数 ───
            with gr.Group():
                gr.Markdown("### 预测参数")
                lookback_slider = gr.Slider(
                    50, 512, 400, step=10,
                    label="回顾步数 (lookback)",
                    info="模型参考的历史K线数量，建议 ≤ 512",
                )
                pred_len_slider = gr.Slider(
                    10, 200, 120, step=5,
                    label="预测步数 (pred_len)",
                    info="预测未来的K线数量",
                )
                temperature_slider = gr.Slider(
                    0.1, 2.0, 1.0, step=0.1,
                    label="温度 (T)",
                    info="越高预测越多样化",
                )
                top_p_slider = gr.Slider(
                    0.5, 1.0, 0.9, step=0.05,
                    label="Top-P",
                    info="核采样概率阈值",
                )
                sample_count_slider = gr.Slider(
                    1, 10, 1, step=1,
                    label="采样次数 (sample_count)",
                    info="多次采样取平均，提高稳定性",
                )

            predict_btn = gr.Button("🚀 开始预测", variant="primary", size="lg")

        with gr.Column(scale=2):
            # ─── 输出 ───
            status_text = gr.Markdown("### 就绪，配置参数后点击「开始预测」")
            plot_output = gr.Plot(label="预测图表")
            with gr.Accordion("预测数据表格 (前20行)", open=False):
                table_output = gr.HTML()

    # ─── 事件 ───
    predict_btn.click(
        fn=run_prediction,
        inputs=[
            model_selector, csv_upload, use_sample_cb,
            lookback_slider, pred_len_slider,
            temperature_slider, top_p_slider, sample_count_slider,
        ],
        outputs=[plot_output, table_output, status_text],
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
