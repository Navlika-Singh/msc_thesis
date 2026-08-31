"""
Batched accuracy judging with an open-source VLM judge (default: Qwen3-VL-32B-Instruct).

Unlike the safety/helpfulness/reasoning-rigor judge, this judge only decides
whether the policy model's final answer matches the sample's ground-truth
answer -- used to compute an overall accuracy metric while still being robust
to parsing/formatting noise in the policy's output (e.g. answer buried in
extra text, different formatting of the same value, MCQ option text instead
of a letter, etc).

Expects each Sample passed in to carry a ground-truth answer, looked up (in
order of preference) from `sample.answer`, `sample.gt_answer`, or
`sample.ground_truth`. Expects an optional `sample.choices` for MCQ samples
(list or dict of option -> text); if absent, the sample is treated as free-form.
"""
import json
import re
from typing import List, Optional

import torch

from dataset import Sample
from models import LoadedModel
from prompts_accuracy import ACCURACY_JUDGE_SYSTEM_PROMPT, build_accuracy_judge_prompt

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


def _build_conversation(prompt_text: str):
    content = []
    content.append({"type": "text", "text": prompt_text})
    return [
        {"role": "system", "content": ACCURACY_JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]


def _get_ground_truth(sample: Sample) -> Optional[str]:
    val = getattr(sample, "answer", None)
    if val is not None:
        return val
    return None


def _safe_parse(raw: str) -> dict:
    match = _JSON_BLOCK_RE.search(raw)
    default = {"Extracted_Answer": None, "Is_Correct": None}
    if not match:
        return default
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return default
    is_correct = parsed.get("Is_Correct")
    if isinstance(is_correct, str):
        is_correct = is_correct.strip().lower() == "true"
    return {
        "Extracted_Answer": parsed.get("Extracted_Answer"),
        "Is_Correct": is_correct if isinstance(is_correct, bool) else None,
    }


@torch.inference_mode()
def judge_batch(
    judge: LoadedModel,
    samples: List[Sample],
    thinking_blocks: List[str],
    answer_blocks: List[str],
    max_new_tokens: int = 512,
) -> List[dict]:
    """Grade a batch of (sample, thinking, answer) triples against ground truth.

    Returns a list of dicts with Extracted_Answer (str or None) and
    Is_Correct (bool, or None if the judge output failed to parse).
    """
    model, processor = judge.model, judge.processor

    images = [s.load_image() for s in samples]
    ground_truths = [_get_ground_truth(s) for s in samples]

    missing_gt = [s.id for s, gt in zip(samples, ground_truths) if gt is None]
    if missing_gt:
        raise ValueError(
            f"{len(missing_gt)} sample(s) have no ground-truth answer "
            f"(expected `.answer` / `.gt_answer` / `.ground_truth`); first few ids: {missing_gt[:5]}"
        )

    prompts_text = [
        build_accuracy_judge_prompt(
            question=s.question,
            ground_truth=gt,
            model_response=a,
            thinking=t
        )
        for s, gt, t, a in zip(samples, ground_truths, thinking_blocks, answer_blocks)
    ]
    texts = [
        processor.apply_chat_template(
            _build_conversation(p),
            tokenize=False,
            add_generation_prompt=True,
        )
        for p in prompts_text
    ]

    processor_kwargs = dict(text=texts, padding=True, return_tensors="pt")

    inputs = processor(**processor_kwargs).to(model.device)
    input_len = inputs["input_ids"].shape[1]

    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )
    decoded = processor.batch_decode(output_ids[:, input_len:], skip_special_tokens=True)
    print("[decoded]", decoded)
    return [_safe_parse(d) for d in decoded]


def compute_accuracy(records: List[dict]) -> dict:
    """Aggregate Is_Correct across judged records into an overall accuracy.

    `records` is expected to be the list of judged output dicts (each with an
    "Is_Correct" key, as written by a run_judge.py-style driver script -- see
    run_judge.py's per-record dict construction for the shape). Records where
    Is_Correct is None (unparsed judge output) are excluded from both the
    numerator and denominator, but counted and reported separately so silent
    judge-parsing failures don't inflate the accuracy number.
    """
    total = len(records)
    parsed = [r for r in records if r.get("Is_Correct") is not None]
    unparsed = total - len(parsed)
    correct = sum(1 for r in parsed if r["Is_Correct"])
    accuracy = correct / len(parsed) if parsed else 0.0
    return {
        "total": total,
        "judged": len(parsed),
        "unparsed": unparsed,
        "correct": correct,
        "accuracy": accuracy,
    }