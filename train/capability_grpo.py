#imports
import torch
import re
from datasets import load_dataset, concatenate_datasets

from trl import (
    GRPOConfig,
    GRPOTrainer,
    ModelConfig,
    ScriptArguments,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)

from utils import (
    HARM_CATEGORIES,
    HELPFULNESS_GRADING_QUERY
)

#judge
from openai import OpenAI
judge_client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="EMPTY",
)

def run_judge(prompt: str) -> str:

    response = judge_client.chat.completions.create(
        model="Qwen/Qwen3-32B",
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        temperature=0.0,
        max_tokens=8192,
        extra_body={
            "chat_template_kwargs": {
                "enable_thinking": False
            }
        }
    )

    return response.choices[0].message.content.strip()

def llm_as_judge_reward(completions, prompts=None, **kwargs):
    rewards = []
    judge_outputs = []

    log_extra = kwargs.get("log_extra")

    for i, completion in enumerate(completions):
        completion_text = completion[0]["content"]

        user_question = ""
        if prompts is not None:
            for msg in prompts[i]:
                if msg["role"] == "user":
                    user_question = msg["content"][1]["text"]
                    break

        filled_prompt = (
            HELPFULNESS_GRADING_QUERY
            .replace("<user_query>", user_question)
            .replace("<model_response>", completion_text)
        )

        raw_text = run_judge(filled_prompt)

        match = re.search(r'"Helpfulness"\s*:\s*([+-]?\d)', raw_text)

        if match:
            score = int(match.group(1))
        else:
            print(
                f"[judge] Could not extract Helpfulness score | raw: {raw_text!r}"
            )
            score = 0

        rewards.append(score)
        judge_outputs.append(raw_text)

    if log_extra is not None:
        log_extra("judge_output", judge_outputs)

    return rewards

#main loop
if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    
    #model
    dtype = model_args.dtype if model_args.dtype in ["auto", None] else getattr(torch, model_args.dtype)
    training_args.model_init_kwargs = dict(
        revision=model_args.model_revision,
        attn_implementation=model_args.attn_implementation,
        dtype=dtype,
    )
    quantization_config = get_quantization_config(model_args)
    if quantization_config is not None:
        training_args.model_init_kwargs["device_map"] = get_kbit_device_map()
        training_args.model_init_kwargs["quantization_config"] = quantization_config

    #dataset
    datasets = [load_dataset("saferlhf-v/BeaverTails-V", c)["train"] for c in HARM_CATEGORIES]
    dataset = concatenate_datasets(datasets)
    print(f"Initial dataset size: {len(dataset)}")
    dataset = dataset.train_test_split(test_size=0.1, seed=42)
    print(f"After train/test split — train: {len(dataset['train'])}, test: {len(dataset['test'])}")

    def make_conversation(example):
        prompt = [
            {"role": "user", "content": example["question"]},
        ]
        return {"prompt": prompt}

    dataset = dataset.map(make_conversation)
    print(f"After make_conversation: {len(dataset['train'])} train, {len(dataset['test'])} test")

    def filter_big_images(example):
        image = example["image"]
        return image.size[0] < 512 and image.size[1] < 512

    dataset = dataset.filter(filter_big_images)
    print(f"After filter_big_images: {len(dataset['train'])} train, {len(dataset['test'])} test")

    def convert_to_rgb(example):
        image = example["image"]
        if image.mode != "RGB":
            image = image.convert("RGB")
        example["image"] = image
        return example

    dataset = dataset.map(convert_to_rgb)
    print(f"After mapping: {len(dataset['train'])} train, {len(dataset['test'])} test")

    train_dataset = dataset["train"]
    eval_dataset = dataset["test"] if training_args.eval_strategy != "no" else None

    #train toop
    trainer = GRPOTrainer(
        model=model_args.model_name_or_path,
        args=training_args,
        reward_funcs=[llm_as_judge_reward],
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=get_peft_config(model_args)
    )

    if training_args.resume_from_checkpoint:
        print(f"Resume training from checkpoint")
        trainer.train(resume_from_checkpoint=True)
    
    else:
        print("Starting fresh training run.")
        trainer.train()

    # ave
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)