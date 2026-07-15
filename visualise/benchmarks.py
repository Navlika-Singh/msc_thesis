import json
from pathlib import Path

import gradio as gr
import pandas as pd
from PIL import Image

JSONL_FILE = "/projects/u6lm/ns1324/msc_thesis/results/benchmarking/beavertails-v/qwen2-vl-7b-instruct"
IMAGE_DIR = "/projects/u6lm/ns1324/msc_thesis/data/beavertails-v"

records = []

with open(JSONL_FILE, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            records.append(json.loads(line))

df = pd.DataFrame(records)

print(f"[DEBUG] Loaded {len(df)} samples.")

CATEGORY_OPTIONS = ["All"] + sorted(df["category"].unique().tolist())

DISPLAY_COLUMNS = [
    "index",
    "question",
    "category",
    "image_severity",
    "response_1",
    "response_2",
    "response_1_from",
    "response_2_from",
    "more_helpful_response_id",
    "is_response_1_safe",
    "is_response_2_safe",
    "safer_response_id",
    "response_1_harmless_rate",
    "response_2_harmfless_rate",
    "response_1_helpful_rate",
    "response_2_helpful_rate",
]

def get_filtered_df(category):

    if category == "All":
        return df

    return df[df["category"] == category].reset_index(drop=True)


def update_slider(category):

    filtered_df = get_filtered_df(category)

    return gr.update(
        minimum=0,
        maximum=max(len(filtered_df) - 1, 0),
        value=0,
    )


def load_sample(category, idx):

    filtered_df = get_filtered_df(category)

    if len(filtered_df) == 0:
        return None, "No samples found.", ""

    idx = int(idx)
    idx = min(idx, len(filtered_df) - 1)

    row = filtered_df.iloc[idx]

    image_path = Path(IMAGE_DIR) / f"{int(row['index']):04d}.png"

    if image_path.exists():
        image = Image.open(image_path).convert("RGB")
    else:
        image = None

    metadata = ""

    for key in DISPLAY_COLUMNS:
        metadata += f"### {key}\n"
        metadata += f"{row[key]}\n\n"

    model_response = row["model_response"]

    return image, metadata, model_response


with gr.Blocks(title="BeaverTails-V Viewer") as demo:

    gr.Markdown("# BeaverTails-V Inference Viewer")

    with gr.Row():

        category = gr.Radio(
            choices=CATEGORY_OPTIONS,
            value="All",
            label="Category",
        )

    with gr.Row():

        sample_idx = gr.Slider(
            minimum=0,
            maximum=len(df) - 1,
            value=0,
            step=1,
            label="Sample Index",
        )

    with gr.Row():

        image = gr.Image(
            label="Image",
            type="pil",
            height=500,
        )

        metadata = gr.Markdown(
            label="Metadata",
        )

    with gr.Row():

        model_response = gr.Textbox(
            label="Model Response",
            lines=12,
            interactive=False,
        )

    category.change(
        fn=update_slider,
        inputs=category,
        outputs=sample_idx,
    )

    category.change(
        fn=load_sample,
        inputs=[category, sample_idx],
        outputs=[image, metadata, model_response],
    )

    sample_idx.change(
        fn=load_sample,
        inputs=[category, sample_idx],
        outputs=[image, metadata, model_response],
    )

    demo.load(
        fn=load_sample,
        inputs=[category, sample_idx],
        outputs=[image, metadata, model_response],
    )

demo.launch()