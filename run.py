import os
import argparse
import yaml
import json

from dataset import get_dataset
from models import get_model

from qwen_vl_utils import process_vision_info
from torch.utils.data import DataLoader

def main():

    parser = argparse.ArgumentParser(description="Benchmarking models.")
    parser.add_argument(
        "--config",
        type=str,
        default="/projects/u6lm/ns1324/msc_thesis/configs/beavertails-v/qwen2-vl-7b-instruct.yaml",
        help="Path to YAML config file.",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    #load dataset
    dataset = get_dataset(
        dataset_id=cfg["dataset"]["dataset_id"],
        split=cfg["dataset"]["split"],
    )
    batch_size = cfg["runtime"]["batch_size"]
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda x:x,
    )

    #load model
    model, processor = get_model(
        cfg["model"]["key"]
    )

    outputs = os.path.join(cfg["output_dir"], cfg["model"]["key"])
    with open(outputs, "w") as f:
        idx = 0
        for batch in loader:

            messages_batch = []
            for sample in batch:
                messages_batch.append( 
                    [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image",
                                    "image": sample["image"],
                                },
                                {
                                    "type": "text",
                                    "text": sample["question"],
                                },
                            ],
                        }
                    ]
                )

            texts = [
                processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                )
                for messages in messages_batch
            ]

            image_inputs = []
            video_inputs = []

            for messages in messages_batch:
                image, video = process_vision_info(messages)
                image_inputs.extend(image)
                video_inputs.extend(video)

            inputs = processor(
                text=texts,
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )

            inputs = inputs.to("cuda")

            generated_ids = model.generate(
                **inputs,
                **cfg["generation"]
            )

            generated_ids_trimmed = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(
                    inputs.input_ids,
                    generated_ids,
                )
            ]

            response = processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

            for sample, response in zip(batch, responses):

                record = {
                        "index": idx,
                        "question": sample["question"],
                        "category": sample["category"],
                        "image_severity": sample["image_severity"],
                        "response_1": sample["response_1"],
                        "response_2": sample["response_2"],
                        "response_1_from": sample["response_1_from"],
                        "response_2_from": sample["response_2_from"],
                        "more_helpful_response_id": sample["more_helpful_response_id"],
                        "is_response_1_safe": sample["is_response_1_safe"],
                        "is_response_2_safe": sample["is_response_2_safe"],
                        "safer_response_id": sample["safer_response_id"],
                        "response_1_harmless_rate": sample["response_1_harmless_rate"],
                        "response_2_harmfless_rate": sample["response_2_harmless_rate"],
                        "response_1_helpful_rate": sample["response_1_helpful_rate"],
                        "response_2_helpful_rate": sample["response_2_helpful_rate"],
                        "model_response": response
                    }
            
                f.write(json.dumps(record) + "\n")
                print(f"[{idx}] {response}")
                idx += 1

            f.flush()


if __name__ == "__main__":
    main()