from typing import List, Optional

import torch

from dataset import Sample
from models import LoadedModel
from prompts import POLICY_SYSTEM_PROMPT
from prompt_strategies import STRATEGIES


def _build_conversation(sample: Sample, strategy, has_image: bool):
    content = []
    if has_image:
        content.append({"type": "image"})
    content.append({"type": "text", "text": STRATEGIES[strategy]["beavertails_v"] + sample.question})
    return [
        {"role": "system", "content": [{"type": "text", "text": POLICY_SYSTEM_PROMPT}]},
        {"role": "user", "content": content},
    ]


@torch.inference_mode()
def generate_batch(
    policy: LoadedModel,
    samples: List[Sample],
    strategy: str = "direct",
    max_new_tokens: int = 1024,
    do_sample: bool = False,
    temperature: float = 1.0,
    num_return_sequences: int = 1,
) -> List[List[str]]:
    """Generate policy responses for a batch of samples.

    Returns a list (len == len(samples)) of lists (len == num_return_sequences)
    of raw decoded strings containing <thinking>...</thinking><answer>...</answer>.
    """
    model, processor = policy.model, policy.processor

    images = [s.load_image() for s in samples]

    texts = [
        processor.apply_chat_template(
            _build_conversation(s, strategy, has_image=img is not None),
            add_generation_prompt=True,
        )
        for s, img in zip(samples, images)
    ]

    has_any_image = all(img is not None for img in images)
    processor_kwargs = dict(text=texts, padding=True, return_tensors="pt")
    if has_any_image:
        # Vision processors expect None to be filtered out per-sample; text-only
        # samples in a mixed batch are not supported by most VLM processors, so
        # in practice keep batches homogeneous (all-image or all-text).
        processor_kwargs["images"] = images

    inputs = processor(**processor_kwargs).to(model.device)
    input_len = inputs["input_ids"].shape[1]

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        num_return_sequences=num_return_sequences,
    )
    if do_sample:
        gen_kwargs["temperature"] = temperature

    output_ids = model.generate(**inputs, **gen_kwargs)
    new_tokens = output_ids[:, input_len:]
    decoded = processor.batch_decode(new_tokens, skip_special_tokens=True)

    # Reshape flat [batch * num_return_sequences] -> [batch][num_return_sequences]
    results = []
    for i in range(len(samples)):
        start = i * num_return_sequences
        results.append(decoded[start : start + num_return_sequences])
    return results


def parse_thinking_answer(raw: str) -> dict:
    """Best-effort split of a raw generation into thinking/answer blocks."""
    import re

    think_match = re.search(r"<thinking>(.*?)</thinking>", raw, re.DOTALL)
    answer_match = re.search(r"<answer>(.*?)</answer>", raw, re.DOTALL)
    return {
        "thinking": think_match.group(1).strip() if think_match else "",
        "answer": answer_match.group(1).strip() if answer_match else raw.strip(),
        "raw": raw,
    }
