# SC-GRPO for Multimodal LLM Safety Alignment

This repository contains the training and evaluation code for **SC-GRPO** and related GRPO variants for safety and capability alignment of Multimodal Large Language Models (MLLMs).

## Training

All training scripts are available under:

```bash
cd train/
```

### SC-GRPO

To train a model using SC-GRPO, use:

```bash
bash scgrpo.sh
```

Before running, update the relevant entries in `scgrpo.sh` for your setup.

The main parameters you may need to change are:

```bash
export TRITON_CACHE_DIR="/path/to/triton_cache"
export HF_HOME="/path/to/hf_cache"
export CUDA_VISIBLE_DEVICES=0
```

and:

```bash
--num_processes 1
--model_name_or_path Qwen/Qwen2-VL-7B-Instruct
--output_dir /path/to/output
```

If using the default experimental configuration, the remaining SC-GRPO hyperparameters can be kept unchanged.

The main SC-GRPO specific parameters are:

```bash
--safety_threshold 2.0
--lambda_init 1.0
--lambda_lr 0.1
--lambda_max 20.0
```

Adjust the GPU configuration, cache paths, model name, and output directory according to your compute environment.

### Other GRPO Variants

The other GRPO variants can be trained in the same way using:

```bash
bash safety_grpo.sh
bash capability_grpo.sh
bash multiobjective_grpo.sh
```

Update the relevant compute, model, and output settings in each script before running.

## Evaluation

All evaluation scripts are available under:

```bash
cd evaluation/
```

Evaluation is performed in two stages:

1. Generate model responses using `run_eval.py`.
2. Judge the generated responses using the appropriate evaluation script.

### Generate Responses

For both safety and capability benchmarks:

```bash
python run_eval.py \
    --data PKU-Alignment/BeaverTails-V \
    --split evaluation \
    --policy qwen2.5-vl-7b-instruct \
    --judge qwen3-vl-32b-instruct \
    --batch-size 8 \
    --out outputs/results.jsonl
```

Update `--data`, `--split`, `--policy`, `--judge`, `--batch-size`, and `--out` as required.

## Safety Evaluation

For safety benchmarks, evaluate the generated responses using:

```bash
python run_judge.py \
    --generations outputs/results.jsonl \
    --data PKU-Alignment/BeaverTails-V \
    --split evaluation \
    --judge qwen3-vl-32b-instruct \
    --batch-size 8 \
    --out outputs/results_judged.jsonl
```

## Capability Evaluation

For capability benchmarks with ground-truth answers, use:

```bash
python run_judge_accuracy.py \
    --generations outputs/results.jsonl \
    --data your/dataset-with-ground-truth \
    --split evaluation \
    --judge qwen3-vl-32b-instruct \
    --batch-size 8 \
    --out outputs/results_judged_accuracy.jsonl
```

The same evaluation pipeline can be used for models trained with SC-GRPO, Safety-GRPO, Capability-GRPO, Multi-Objective GRPO, and baseline models.

## Contact

For any questions, please contact the author at **[navlikas1309@gmail.com](mailto:navlikas1309@gmail.com)**.
