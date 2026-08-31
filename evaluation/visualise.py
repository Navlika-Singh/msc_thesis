import json
from pathlib import Path

import gradio as gr
import pandas as pd
from PIL import Image

RESULTS = "/rds/general/user/ns1324/home/msc_thesis/safer-toolkit-eval/outputs/vlsbench_qwen2-vl-7b-instruct_results.jsonl"
JUDGED = "/rds/general/user/ns1324/home/msc_thesis/safer-toolkit-eval/outputs/vlsbench_qwen2-vl-7b-instruct_results_judged.jsonl"
SUMMARY = "/rds/general/user/ns1324/home/msc_thesis/safer-toolkit-eval/outputs/vlsbench_qwen2-vl-7b-instruct_results_judged_summary.json"

IMAGE_DIR = "/rds/general/user/ns1324/home/msc_thesis/data/beavertails-v"


def load_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f]


raw = {x["id"]: x for x in load_jsonl(RESULTS)}
judged = load_jsonl(JUDGED)

for x in judged:
    if x["id"] in raw:
        x["thinking"] = raw[x["id"]].get("thinking", "")
        x["answer"] = raw[x["id"]].get("answer", "")

df = pd.DataFrame(judged)

with open(SUMMARY) as f:
    summary = json.load(f)


categories = ["All"] + sorted(df.category.unique().tolist())
safety_scores = ["All"] + sorted(df.Answer_Safety_Score.unique().tolist())
help_scores = ["All"] + sorted(df.Answer_Helpfulness_Score.unique().tolist())
rigor_scores = ["All"] + sorted(df.Reasoning_Rigor_Score.unique().tolist())


def get_image(idx):
    sample = current.iloc[idx]
    stem = sample["id"][-4:]

    for ext in [".jpg", ".jpeg", ".png", ".webp"]:
        p = Path(IMAGE_DIR) / (stem + ext)
        print("image path:", p)
        if p.exists():
            return Image.open(p)

    return None


def filter_df(category, safety, helpfulness, rigor, search):

    d = df

    if category != "All":
        d = d[d.category == category]

    if safety != "All":
        d = d[d.Answer_Safety_Score == safety]

    if helpfulness != "All":
        d = d[d.Answer_Helpfulness_Score == helpfulness]

    if rigor != "All":
        d = d[d.Reasoning_Rigor_Score == rigor]

    if search.strip():
        s = search.lower()
        d = d[
            d.question.str.lower().str.contains(s)
            | d.answer.str.lower().str.contains(s)
            | d.thinking.str.lower().str.contains(s)
        ]

    return d.reset_index(drop=True)


current = df.copy()


def refresh(category, safety, helpfulness, rigor, search):

    global current

    current = filter_df(category, safety, helpfulness, rigor, search)

    if len(current) == 0:
        return (
            current,
            0,
            None,
            "",
            "",
            "",
            "",
            "",
        )

    return (
        current,
        0,
        *show(0),
    )


def show(idx):

    idx = max(0, min(idx, len(current) - 1))

    x = current.iloc[idx]

    img = get_image(idx)

    scores = (
        f"Safety: {x.Answer_Safety_Score}\n"
        f"Helpfulness: {x.Answer_Helpfulness_Score}\n"
        f"Reasoning: {x.Reasoning_Rigor_Score}"
    )

    return (
        img,
        x.question,
        x.thinking,
        x.answer,
        scores,
        f"{idx+1}/{len(current)}",
    )


def prev(idx):
    idx = max(0, idx - 1)
    return (idx, *show(idx))


def next_(idx):
    idx = min(len(current) - 1, idx + 1)
    return (idx, *show(idx))


summary_text = json.dumps(summary, indent=2)


with gr.Blocks() as demo:

    gr.Markdown("# Judge Results Viewer")

    gr.Code(summary_text, language="json")

    with gr.Row():
        category = gr.Dropdown(categories, value="All", label="Category")
        safety = gr.Dropdown(safety_scores, value="All", label="Safety")
        helpfulness = gr.Dropdown(help_scores, value="All", label="Helpfulness")
        rigor = gr.Dropdown(rigor_scores, value="All", label="Reasoning")
        search = gr.Textbox(label="Search")

    table = gr.Dataframe(df)

    state = gr.State(0)

    with gr.Row():
        prev_btn = gr.Button("Previous")
        counter = gr.Textbox(label="Index")
        next_btn = gr.Button("Next")

    image = gr.Image(height=400)

    question = gr.Textbox(label="Question", lines=3)
    thinking = gr.Textbox(label="Thinking", lines=12)
    answer = gr.Textbox(label="Answer", lines=6)
    scores = gr.Textbox(label="Scores")

    btn = gr.Button("Apply Filters")

    btn.click(
        refresh,
        [category, safety, helpfulness, rigor, search],
        [
            table,
            state,
            image,
            question,
            thinking,
            answer,
            scores,
            counter,
        ],
    )

    prev_btn.click(
        prev,
        state,
        [
            state,
            image,
            question,
            thinking,
            answer,
            scores,
            counter,
        ],
    )

    next_btn.click(
        next_,
        state,
        [
            state,
            image,
            question,
            thinking,
            answer,
            scores,
            counter,
        ],
    )

    demo.load(
        lambda: (
            df,
            0,
            *show(0),
        ),
        outputs=[
            table,
            state,
            image,
            question,
            thinking,
            answer,
            scores,
            counter,
        ],
    )

demo.launch()