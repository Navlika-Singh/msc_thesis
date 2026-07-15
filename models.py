from dataclasses import dataclass, field
from typing import Optional

import torch
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor

REGISTRY = {
    "qwen2-vl-7b-instruct": {
        "cls": Qwen2VLForConditionalGeneration,
        "path": "Qwen/Qwen2-VL-7B-Instruct",
    }
}

def get_model(key):

    cfg = REGISTRY[key]
    
    if "qwen2-vl" in key:
        print(f"[DEBUG] Loading model family {key} and model spec {cfg["path"]}")
        model = cfg["cls"].from_pretrained(
            cfg["path"],
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

        print(f"[DEBUG] Loading processor")
        if "min_pixels" in cfg and "max_pixels" in cfg:
            print(f"[DEBUG] min pixels: {cfg["min_pixels"]} and max pixels: {cfg["max_pixels"]}")
            processor = AutoProcessor.from_pretrained(
            cfg["path"],
            min_pixels=cfg["min_pixels"],
            max_pixels=cfg["max_pixels"]
            )
        else:
            processor = AutoProcessor.from_pretrained(
                cfg["path"]
            )
    
        return model, processor 
    else:
        raise KeyError(f"Invalid model key {key}. Valid model keys are: {list(REGISTRY)}")


    