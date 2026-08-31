"""
Model registry for policy (responder) VLMs and the judge VLM.

Follows the same REGISTRY / get_model pattern used elsewhere in the codebase:
a plain dict keyed by model name -> {"cls": ..., "path": ...}, with a thin
family dispatch in get_model() for anything that needs non-default kwargs
(min/max pixels, dtype, etc).
"""
from dataclasses import dataclass
from typing import Optional

import torch
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    BitsAndBytesConfig,
    LlavaNextProcessor, LlavaNextForConditionalGeneration
)

from peft import PeftModel
# ---------------------------------------------------------------------------
# Policy (responder) models under evaluation.
# ---------------------------------------------------------------------------
POLICY_REGISTRY = {
    "qwen2.5-vl-3b-instruct": {
        "cls": Qwen2_5_VLForConditionalGeneration,
        "path": "/vol/bitbucket/ns1324/msc_thesis/models/Qwen2.5-VL-3B-Instruct",
    },
    "llava1.6-7b": {
        "cls": LlavaNextForConditionalGeneration,
        "path": "/vol/bitbucket/ns1324/msc_thesis/models/llava-v1.6",
    },
    "safe-rlhf-v-llava": {
        "cls": LlavaNextForConditionalGeneration,
        "path": "/vol/bitbucket/ns1324/msc_thesis/models/LLaVA-NeXT_Safe_RLHF-V",
    },
    "safe-rlhf-v-qwen": {
        "cls": Qwen2VLForConditionalGeneration,
        "path": "/vol/bitbucket/ns1324/msc_thesis/models/Qwen_Safe_RLHF-V",
    },
    "qwen2.5-vl-7b-instruct": {
        "cls": Qwen2_5_VLForConditionalGeneration,
        "path": "Qwen/Qwen2.5-VL-7B-Instruct",
    },
    "qwen2-vl-7b-instruct": {
        "cls": Qwen2VLForConditionalGeneration,
        "path": "Qwen/Qwen2-VL-7B-Instruct",
    },
    "qwen2.5-vl-3b-helpfulness-lora": {
        "cls": Qwen2_5_VLForConditionalGeneration,
        "path": "/vol/bitbucket/ns1324/msc_thesis/models/Qwen2.5-VL-3B-Instruct",
        "adapter": "/vol/bitbucket/ns1324/msc_thesis/models/grpo-Qwen2.5-VL-3B-Instruct_base_HELPFULNESSalone/checkpoint-1220",
    },
    "qwen2-vl-7b-helpfulness-lora": {
        "cls": Qwen2VLForConditionalGeneration,
        "path": "Qwen/Qwen2-VL-7B-Instruct",
        "adapter": "/vol/bitbucket/ns1324/msc_thesis/grpo-Qwen2-VL-7B-Instruct_base_HELPFULNESSalone/checkpoint-600",
    },
    "llava1.6-7b-helpfulness-lora": {
        "cls": LlavaNextForConditionalGeneration,
        "path": "/vol/bitbucket/ns1324/msc_thesis/models/llava-v1.6",
        "adapter": "/vol/bitbucket/ns1324/msc_thesis/models/grpo-llava-v1.6-7b_base_HELPFULNESSalone/checkpoint-500",
    },
    "qwen2.5-vl-3b-safety-lora": {
        "cls": Qwen2_5_VLForConditionalGeneration,
        "path": "/vol/bitbucket/ns1324/msc_thesis/models/Qwen2.5-VL-3B-Instruct",
        "adapter": "/vol/bitbucket/ns1324/msc_thesis/models/grpo-Qwen2.5-VL-3B-Instruct_base_SAFETYalone/checkpoint-770",
    },
    "qwen2-vl-7b-safety-lora": {
        "cls": Qwen2VLForConditionalGeneration,
        "path": "Qwen/Qwen2-VL-7B-Instruct",
        "adapter": "/vol/bitbucket/ns1324/msc_thesis/grpo-Qwen2-VL-7B-Instruct_base_SAFETYalone/checkpoint-700",
    },
    "llava1.6-7b-safety-lora": {
        "cls": LlavaNextForConditionalGeneration,
        "path": "/vol/bitbucket/ns1324/msc_thesis/models/llava-v1.6",
        "adapter": "/vol/bitbucket/ns1324/msc_thesis/models/grpo-llava-v1.6-7b_base_SAFETYalone/checkpoint-500",
    },
    "qwen2.5-vl-3b-helpfulness-safety-lora": {
        "cls": Qwen2_5_VLForConditionalGeneration,
        "path": "/vol/bitbucket/ns1324/msc_thesis/models/Qwen2.5-VL-3B-Instruct",
        "adapter": "/vol/bitbucket/ns1324/msc_thesis/models/grpo-Qwen2.5-VL-3B-Instruct_helpfull_safety_MOGRPO/checkpoint-910",
    },
    "qwen2.5-vl-3b-helpfulness-safety-lora-nomogrpo": {
        "cls": Qwen2_5_VLForConditionalGeneration,
        "path": "/vol/bitbucket/ns1324/msc_thesis/models/Qwen2.5-VL-3B-Instruct",
        "adapter": "/vol/bitbucket/ns1324/msc_thesis/models/grpo-Qwen2.5-VL-3B-Instruct_helpfull_safety/checkpoint-420",
    },
    "qwen2-vl-7b-helpfulness-safety-lora": {
        "cls": Qwen2VLForConditionalGeneration,
        "path": "Qwen/Qwen2-VL-7B-Instruct",
        "adapter": "/vol/bitbucket/ns1324/msc_thesis/models/grpo-Qwen2-VL-7B-Instruct_helpfull_safety_MOGRPO/checkpoint-1000",
    },
    "llava1.6-7b-helpfulness-safety-lora": {
        "cls": LlavaNextForConditionalGeneration,
        "path": "/vol/bitbucket/ns1324/msc_thesis/models/llava-v1.6",
        "adapter": "/vol/bitbucket/ns1324/msc_thesis/models/grpo-llava-v1.6-7b_helpfull_safety_MOGRPO/checkpoint-500",
    },
    "cgrpo-Qwen2.5-VL-2B-Instruct_lambda_lr0.1_lambda_init0.1_lambda_max20_safety_threshold-2_helpstatic": {
        "cls": Qwen2_5_VLForConditionalGeneration,
        "path": "/vol/bitbucket/ns1324/msc_thesis/models/Qwen2.5-VL-3B-Instruct",
        "adapter": "/vol/bitbucket/ns1324/msc_thesis/models/cgrpo-Qwen2.5-VL-2B-Instruct_lambda_lr0.1_lambda_init0.1_lambda_max20_safety_threshold-2_helpstatic/checkpoint-1320"
    },
    "qwen2.5-vl-3b-helpfulness-safety-cgrpo": {
        "cls": Qwen2_5_VLForConditionalGeneration,
        "path": "/vol/bitbucket/ns1324/msc_thesis/models/Qwen2.5-VL-3B-Instruct",
        "adapter": "/vol/bitbucket/ns1324/msc_thesis/models/cgrpo-Qwen2.5-VL-2B-Instruct_lambda_lr0.1_lambda_init1_lambda_max2_safety_threshold-2/checkpoint-1580",
    },
    "qwen2.5-vl-3b-helpfulness-safety-cgrpo-dynamic": {
        "cls": Qwen2_5_VLForConditionalGeneration,
        "path": "/vol/bitbucket/ns1324/msc_thesis/models/Qwen2.5-VL-3B-Instruct",
        "adapter": "/vol/bitbucket/ns1324/msc_thesis/models/cgrpo-Qwen2.5-VL-2B-Instruct_lambda_lr0.1_lambda_init0.1_lambda_max20_safety_threshold-2_helpdynamic/checkpoint-1680",
    },
    "qwen2-vl-7b-helpfulness-safety-cgrpo-dynamic": {
        "cls": Qwen2VLForConditionalGeneration,
        "path": "Qwen/Qwen2-VL-7B-Instruct",
        "adapter": "/vol/bitbucket/ns1324/msc_thesis/cgrpo-Qwen2-VL-7B-Instruct_lambda_lr0.1_lambda_init0.1_lambda_max20_safety_threshold-2_helpdynamic/checkpoint-1100",
    },
    "llava1.6-7b-helpfulness-safety-cgrpo-dynamic": {
        "cls": LlavaNextForConditionalGeneration,
        "path": "/vol/bitbucket/ns1324/msc_thesis/models/llava-v1.6",
        "adapter": "/vol/bitbucket/ns1324/msc_thesis/models/cgrpo-llava-v1.6-7b_lambda_lr0.1_lambda_init0.1_lambda_max20_safety_threshold-2_helpdynamic/checkpoint-1000",
    },
    "qwen2.5-vl-3b-helpfulness-safety-cgrpo-dual": {
        "cls": Qwen2_5_VLForConditionalGeneration,
        "path": "/vol/bitbucket/ns1324/msc_thesis/models/Qwen2.5-VL-3B-Instruct",
        "adapter": "/vol/bitbucket/ns1324/msc_thesis/models/cgrpo-Qwen2.5-VL-2B-Instruct_lambda_lr0.1_lambda_init0.1_lambda_max20_safety_threshold-2_helpdual/checkpoint-1420",
    },
    "qwen2.5-vl-3b-helpfulness-safety-cgrpo-nocapweight": {
        "cls": Qwen2_5_VLForConditionalGeneration,
        "path": "/vol/bitbucket/ns1324/msc_thesis/models/Qwen2.5-VL-3B-Instruct",
        "adapter": "/vol/bitbucket/ns1324/msc_thesis/models/cgrpo-Qwen2.5-VL-2B-Instruct_lambda_lr0.1_lambda_init1_lambda_max20_safety_threshold-2/checkpoint-1560",
    },
    "qwen2.5-vl-3b-helpfulness-safety-cgrpo-capweight": {
        "cls": Qwen2_5_VLForConditionalGeneration,
        "path": "/vol/bitbucket/ns1324/msc_thesis/models/Qwen2.5-VL-3B-Instruct",
        "adapter": "/vol/bitbucket/ns1324/msc_thesis/models/cgrpo-Qwen2.5-VL-2B-Instruct_lambda_lr1.0_lambda_init1_lambda_max20_safety_threshold-2_help20/checkpoint-780",
    },

}

# ---------------------------------------------------------------------------
# Judge models. Default is the best open-weight VLM at <=32B: Qwen3-VL-32B-
# Instruct (this is also what the SaFeR-ToolKit paper itself uses as its
# reward/judge model in Appendix C.3, Fig. 5 -- the eval-time judge in the
# paper is GPT-5-mini, a closed model, which we swap out here for an
# open-source judge to keep the whole pipeline self-hosted).
# ---------------------------------------------------------------------------
JUDGE_REGISTRY = {
    "qwen3-vl-32b-instruct-4bit": {
        # Qwen3-VL ships via AutoModelForImageTextToText / trust_remote_code
        # rather than a dedicated Qwen3VLForConditionalGeneration class as of
        # this writing; AutoProcessor + AutoModelForImageTextToText handles
        # both cases transparently.
        "cls": AutoModelForImageTextToText,
        "path": "/vol/bitbucket/ns1324/msc_thesis/models/Qwen3-VL-32B-Instruct",
        "trust_remote_code": True,
        "min_pixels": 32*32*1,
        "max_pixels": 32*32*1
    },
    # Fallback judge if you don't have the VRAM for 32B (~65GB in bf16).
    "qwen2.5-vl-32b-instruct": {
        "cls": Qwen2_5_VLForConditionalGeneration,
        "path": "Qwen/Qwen2.5-VL-32B-Instruct",
    },
}

DEFAULT_JUDGE = "qwen3-vl-32b-instruct"


def get_model(key: str, registry: dict, min_pixels: Optional[int] = None, max_pixels: Optional[int] = None):
    """Load a (model, processor) pair from a registry by key.

    Mirrors the dispatch style of the existing get_model() helper: family is
    inferred from the key, and family-specific kwargs (min/max pixels,
    trust_remote_code) are applied as needed. Models are loaded with
    device_map="auto" so batched generation just works across whatever GPUs
    are visible.
    """
    if key not in registry:
        raise KeyError(f"Invalid model key {key!r}. Valid keys are: {list(registry)}")

    cfg = registry[key]
    trust_remote_code = cfg.get("trust_remote_code", False)

    print(f"[DEBUG] Loading model {key} ({cfg['path']})")

    if "4bit" in key:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True
        )

        model = cfg["cls"].from_pretrained(
            cfg["path"],
            quantization_config=quantization_config,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=trust_remote_code,
        )
    else:
        model = cfg["cls"].from_pretrained(
            cfg["path"],
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=trust_remote_code,
        )

    if "adapter" in cfg:
        print(f"[DEBUG] Loading LoRA adapter from {cfg['adapter']}")
        model = PeftModel.from_pretrained(
            model,
            cfg["adapter"],
            is_trainable=False,
        )
    model.eval()

    print(f"[DEBUG] Loading processor for {key}")
    processor_kwargs = {}
    if "min_pixels" in cfg:
        processor_kwargs["min_pixels"] = cfg["min_pixels"]
    if "max_pixels" in cfg:
        processor_kwargs["max_pixels"] = cfg["max_pixels"]
    
    if "llava" in cfg["path"]:
        print("[DEBUG] Llava processor")
        processor = LlavaNextProcessor.from_pretrained(cfg["path"])
    else:
        processor = AutoProcessor.from_pretrained(
            cfg["path"], trust_remote_code=trust_remote_code, **processor_kwargs
        )

    # print(processor.image_processor.min_pixels)
    # print(processor.image_processor.max_pixels)

    # Left padding is required for correct batched generation with decoder-only LMs.
    processor.tokenizer.padding_side = "left"
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    return model, processor


@dataclass
class LoadedModel:
    key: str
    model: object
    processor: object

    @classmethod
    def policy(cls, key: str, min_pixels: Optional[int] = None, max_pixels: Optional[int] = None) -> "LoadedModel":
        model, processor = get_model(key, POLICY_REGISTRY, min_pixels, max_pixels)
        return cls(key, model, processor)

    @classmethod
    def judge(cls, key: str = DEFAULT_JUDGE) -> "LoadedModel":
        model, processor = get_model(key, JUDGE_REGISTRY)
        return cls(key, model, processor)
