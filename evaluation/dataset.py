"""
Dataset loading.

Two sources are supported:

1. JSONL, one record per line:
    {
        "id": "beavertails_0001",
        "benchmark": "BeaverTails-V",   # used to group results, e.g. Table 1
        "image": "data/images/0001.jpg",
        "question": "How do I ...?"
    }
   `image` may be omitted for text-only samples.

2. HF `datasets`, e.g. PKU-Alignment/BeaverTails-V, which ships one config
   per harm category and an `evaluation` split alongside `train`. See
   `load_beavertails_v()` / `HARM_CATEGORIES` below.

=== MMMU-Pro integration ===
Added support for MMMU/MMMU_Pro (standard, 4-option setting). Every change
for this is marked with a "=== MMMU-Pro" comment so it's easy to find/verify:
  - `import ast`
  - `MMMU_PRO_STANDARD_4_CONFIG` constant
  - `Sample.answer` field
  - `load_mmmu_pro_standard()` function
  - a new branch in `get_dataset()`
  - a new `DATASET_CONFIGS["MMMU/MMMU_Pro"]` entry
  - `answer` / `id` support added to `load_eval_dataset()` (both optional,
    via `field_map.get(...)`, so every other dataset is unaffected)
  - `load_mmmu_pro()` convenience wrapper
  - a small default-split fix in `load_for_data_arg()`

This only covers *loading + prompt construction*. Scoring against the
ground-truth `answer` still needs a separate MMMU-Pro-compatible grading
script, per the plan.
"""
import os 
import json
import ast  # === MMMU-Pro: needed to parse the stringified `options` list
from dataclasses import dataclass
from datasets import Dataset, concatenate_datasets, load_dataset
from typing import Iterator, List, Optional

from PIL import Image

HARM_CATEGORIES: list[str] = [
    "psychological_horror_and_dark_themes",
    "psychological_harm_and_manipulation",
    "dangerous_behavior",
    "pornographic_content",
    "harmful_health_content",
    "identity_misuse_and_impersonation",
    "discriminatory_depictions",
    "false_information",
    "privacy_invasion_and_surveillance",
    "financial_and_academic_fraud",
    "sexual_crimes",
    "terrorism_or_extremism",
    "violence_and_physical_harm",
    "deception_in_personal_relationships",
    "sensitive_information_in_key_areas",
    "horror_and_gore",
    "environmental_damage",
    "hacking_or_digital_crime",
    "animal_abuse",
    "insulting_and_harassing_behavior",
]

MMSafetyBench_categories: list[str] = [
    "EconomicHarm",
    "Financial_Advice",
    "Fraud",
    "Gov_Decision",
    "HateSpeech",
    "Health_Consultation",
    "Illegal_Activitiy",
    "Legal_Opinion",
    "Malware_Generation",
    "Physical_Harm",
    "Political_Lobbying",
    "Privacy_Violence",
    "Sex"
]

MMSafetyBench_splits: list[str] = [
    "SD",
    "SD_TYPO",
    "TYPO"
]

SPA_VL_categories: list[str] = [
    "test"
]

MSSBench_categories: list[str] = [
    "chat",
    "embodied"
]

MMMU_categories: list[str] = [
    "Art",
    "Design",
    "Music",
    "Art_Theory",
    "Accounting",
    "Economics",
    "Finance",
    "Manage",
    "Marketing",
    "Biology",
    "Chemistry",
    "Geography",
    "Math",
    "Physics",
    "Basic_Medical_Science",
    "Clinical_Medicine",
    "Diagnostics",
    "Pharmacy",
    "Public_Health",
    "History",
    "Literature",
    "Psychology",
    "Sociology",
    "Agriculture",
    "Architecture_and_Engineering",
    "Computer_Science",
    "Electronics",
    "Energy_Power",
    "Materials",
    "Mechnical_Engineering"
]

MathVista_categories: list[str] = [
    "testmini"
]

MMStar_categories: list[str] = [
    "val"
]

AI2D_categories: list[str] = [
    "test"
]

# Note: MMMU-Pro's "subject" column uses this same 30-subject taxonomy, so
# we don't duplicate the list for it (see load_mmmu_pro_standard() below).

# === MMMU-Pro: config name on the Hub for the "standard, 4-choice" setting.
# The other two configs are "standard (10 options)" and "vision".
MMMU_PRO_STANDARD_4_CONFIG = "standard (4 options)"

@dataclass
class Sample:
    id: str
    benchmark: str
    question: str
    image_path: Optional[str] = None
    image: Optional[Image.Image] = None  # in-memory image, takes priority over image_path
    category: Optional[str] = None       # fine-grained harm category, if available
    # === MMMU-Pro: ground-truth option letter (e.g. "A"). None for
    # benchmarks (like BeaverTails-V) that are judged rather than scored
    # against a fixed answer. Consumed by the (separate) MMMU-Pro scorer.
    answer: Optional[str] = None

    def load_image(self) -> Optional[Image.Image]:
        if self.image is not None:
            return self.image.convert("RGB") if self.image.mode != "RGB" else self.image
        if self.image_path is None:
            return None
        return Image.open(self.image_path).convert("RGB")

def load_mssbench(base_path, categories):
    examples = []
    metadata_path = os.path.join(base_path, "combined.json")
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    for category in categories:
        image_dir = os.path.join(base_path, category)

        entries = metadata[category]

        for sample in entries:
            if category == "chat":
                question = sample["queries"][0]

                # Safe image
                examples.append(
                    {
                        "image": os.path.join(image_dir, sample["safe_image_path"]),
                        "question": question,
                        "category": sample["Type"],
                        "label": "safe",
                    }
                )

                # Unsafe image
                examples.append(
                    {
                        "image": os.path.join(image_dir, sample["unsafe_image_path"]),
                        "question": question,
                        "category": sample["Type"],
                        "label": "unsafe",
                    }
                )
            elif category == "embodied":
                # Safe image
                examples.append(
                    {
                        "image": os.path.join(image_dir, sample["safe"]),
                        "question": sample["safe_instruction"],
                        "category": sample["category"],
                        "label": "safe",
                    }
                )

                # Unsafe image
                examples.append(
                    {
                        "image": os.path.join(image_dir, sample["unsafe"]),
                        "question": sample["unsafe_instruction"],
                        "category": sample["category"],
                        "label": "unsafe",
                    }
                )
            else:
                print("[DEBUG] Category undefined")

    return Dataset.from_list(examples)

def load_vlguard(base_path):
    examples = []
    metadata_path = os.path.join(base_path, "test.json")
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    image_dir = os.path.join(base_path, "test")

    # print(len(metadata))
    # print(len([sample for sample in metadata if sample["safe"] == True]))
    # print(len([sample for sample in metadata if sample["safe"] == False]))

    for sample in metadata:
        # Safe image
        if sample["safe"] == True:
            examples.append(
                {
                    "image": Image.open(os.path.join(image_dir, sample["image"])).convert("RGB"),
                    "question": sample["instr-resp"][0]["safe_instruction"],
                    "category": sample["image"].split("/")[0],
                    "label": "safe",
                }
                )

            # Unsafe image
            examples.append(
                {
                    "image": Image.open(os.path.join(image_dir, sample["image"])).convert("RGB"),
                    "question": sample["instr-resp"][1]["unsafe_instruction"],
                    "category": sample["image"].split("/")[0],
                    "label": "unsafe",
                }
                )
        elif sample["safe"] == False:
            examples.append(
                {
                    "image": Image.open(os.path.join(image_dir, sample["image"])).convert("RGB"),
                    "question": sample["instr-resp"][0]["instruction"],
                    "category": sample["image"].split("/")[0],
                    "label": "unsafe",
                }
                )

    return Dataset.from_list(examples)

def load_ai2d(dataset_id):
    examples = []
    metadata = load_dataset(dataset_id)["test"]

    PROMPT = """Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of A,B,C,D. Think step by step before answering.

    {question}

    A) {choice_1}
    B) {choice_2}
    C) {choice_3}
    D) {choice_4}"""

    for sample in metadata:
        choices = sample["options"]

        prompt = PROMPT.format(
            question=sample["question"],
            choice_1=choices[0],
            choice_2=choices[1],
            choice_3=choices[2],
            choice_4=choices[3],
        )

        examples.append(
            {
                "image": sample["image"],
                "question": prompt,
                "category": None,
                "answer": sample["answer"],
            }
        )

    return Dataset.from_list(examples)

def load_hallusionbench(dataset_id):
    examples = []
    metadata = load_dataset(dataset_id)["test"]

    PROMPT = """Answer the following question by carefully assessing the provided image. The last line of your response should be of the following format: 'ANSWER: [ANSWER]' (without quotes) where [ANSWER] is either 1 (if 'yes/true') or 0 (if 'no/false'). Think step by step before answering.

    {question}"""

    for sample in metadata:
        prompt = PROMPT.format(
            question=sample["question"],
        )

        examples.append(
            {
                "image": sample["image"],
                "question": prompt,
                "category": sample["category"],
                "answer": sample["gt_answer_details"] + "Hence, the answer is: " +  sample["gt_answer"],
            }
        )

    return Dataset.from_list(examples)

# === MMMU-Pro: new loader (same style as load_mssbench/load_vlguard above).
def load_mmmu_pro_standard(num_options: int = 4) -> Dataset:
    """Load the MMMU-Pro 'standard' subset and turn each row into a single
    ready-to-send multiple-choice prompt: question + lettered options +
    the standard MMMU-Pro instruction to answer with just the letter.

    Only a "test" split exists for this config on the Hub, so (like
    load_mssbench/load_vlguard above) the caller's `split` isn't threaded
    through here -- get_dataset() below always resolves to this loader
    regardless of the requested split.

    `options` ships from the Hub as a stringified python list (same
    convention as the base MMMU dataset), so it's parsed with
    ast.literal_eval when needed.
    """
    config_name = (
        MMMU_PRO_STANDARD_4_CONFIG if num_options == 4 else "standard (10 options)"
    )
    raw = load_dataset("MMMU/MMMU_Pro", config_name)["test"]

    letters = [chr(ord("A") + i) for i in range(num_options)]

    examples = []
    for row in raw:
        options = row["options"]
        if isinstance(options, str):
            options = ast.literal_eval(options)

        options_block = "\n".join(
            f"{letter}. {opt}" for letter, opt in zip(letters, options)
        )
        prompt = (
            f"{row['question']}\n{options_block}\n"
            "Answer with the option's letter from the given choices directly."
        )

        examples.append(
            {
                "id": row["id"],
                "question": prompt,
                # Standard setting only needs the primary image here; extra
                # image_2..image_7 slots (rare multi-image questions) aren't
                # threaded through the single-image Sample -- same
                # simplification already made for the base MMMU entry below.
                "image": row.get("image_1"),
                "category": row.get("subject"),
                "answer": row.get("answer"),
            }
        )

    return Dataset.from_list(examples)


def load_samples(jsonl_path: str) -> List[Sample]:
    samples = []
    with open(jsonl_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            samples.append(
                Sample(
                    id=row["id"],
                    benchmark=row.get("benchmark", "default"),
                    question=row["question"],
                    image_path=row.get("image"),
                )
            )
    return samples


def get_dataset(dataset_id="saferlhf-v/BeaverTails-V", split="evaluation", test_size=0.1, seed=42, categories=None):
    """Load a HF dataset for evaluation.

    Generalizes the original BeaverTails-V-specific loader: if the dataset
    ships one HF *config* per harm category (like BeaverTails-V does),
    pass `categories` (defaults to DATASET_CONFIGS[dataset_id]["categories"]
    if registered, else HARM_CATEGORIES for the two known BeaverTails-V
    mirrors) and each config's split is loaded and concatenated. Datasets
    with a single flat config (no per-category split) just get loaded
    directly -- pass `categories=[]` explicitly to force this path for a
    registered dataset.
    """

    if categories is None:
        cfg = DATASET_CONFIGS.get(dataset_id)
        if cfg is not None:
            categories = cfg.get("categories")
        elif dataset_id in ("saferlhf-v/BeaverTails-V", "PKU-Alignment/BeaverTails-V"):
            categories = HARM_CATEGORIES

    print(f"[DEBUG] Loading dataset_id {dataset_id}")
    print(f"[DEBUG] Loading {split}")

    if categories and dataset_id == "saferlhf-v/BeaverTails-V":
        datasets = [load_dataset(dataset_id, c)[split] for c in categories]
        dataset = concatenate_datasets(datasets)
    elif categories and dataset_id == "sqrti/SPA-VL":
        datasets = [
            split_ds
            for c in categories
            for split_ds in load_dataset(dataset_id, c).values()
        ]
        dataset = concatenate_datasets(datasets)
    elif categories and dataset_id == "PKU-Alignment/MM-SafetyBench":
        datasets = [
            split_ds
            for c in categories
            for split_ds in load_dataset(dataset_id, c).values()
        ]
        dataset = concatenate_datasets(datasets)
    elif dataset_id == "Foreshhh/vlsbench":
        dataset = load_dataset(dataset_id)[split]
    elif dataset_id == "kzhou35/mssbench":
        dataset = load_mssbench(
        "/rds/general/user/ns1324/home/msc_thesis/data/mssbench",
        categories,
        )
    elif dataset_id == "ys-zong/VLGuard":
        dataset = load_vlguard(
        "/vol/bitbucket/ns1324/msc_thesis/data/vlguard",
        )
    elif dataset_id == "MMMU/MMMU_Pro":
        # === MMMU-Pro: standard, 4-option setting ===
        dataset = load_mmmu_pro_standard(num_options=4)
    elif dataset_id == "AI4Math/MathVista":
        dataset = load_dataset(dataset_id)[categories[0]]
    elif dataset_id == "Lin-Chen/MMStar":
        dataset = load_dataset(dataset_id)[categories[0]]
    elif dataset_id == "lmms-lab-encoder/ai2d":
        dataset = load_ai2d(dataset_id)
    else:
        dataset = load_dataset(dataset_id)[split]

    # if split == "train":
    #     print(f"[DEBUG] Splitting the dataset {(1 - test_size) * 100}-to-{test_size * 100} train-to-val ratio with seed {seed}")
    #     dataset = dataset.train_test_split(test_size=test_size, seed=seed)
    if split not in ("train", "evaluation", "test", "val", "validation"):
        raise KeyError(f"[ERROR] Unrecognized split '{split}' for {dataset_id}.")

    return dataset


# Registry of known HF eval datasets: per-category configs (if any) + the
# column names to pull question/image/category from. Extend this as you
# wire up more of the papers' benchmarks (MM-SafetyBench, SPA-VL, VLGuard,
# VLSBench, MSSBench, ...) once you've confirmed their exact HF schema --
# BeaverTails-V is filled in below as the reference example.
DATASET_CONFIGS = {
    "PKU-Alignment/BeaverTails-V": {
        "categories": HARM_CATEGORIES,
        "field_map": {"question": "question", "image": "image", "category": "category"},
    },
    "saferlhf-v/BeaverTails-V": {
        "categories": HARM_CATEGORIES,
        "field_map": {"question": "question", "image": "image", "category": "category"},
    },
    "PKU-Alignment/MM-SafetyBench": {
        "categories": MMSafetyBench_categories,
        "field_map": {"question": "question", "image": "image", "category": None, "id":"id"}
    },
    "sqrti/SPA-VL": {
        "categories": SPA_VL_categories,
        "field_map": {"question": "question", "image": "image", "category": "class1",}
    },
    "kzhou35/mssbench": {
        "categories": MSSBench_categories,
        "field_map": {"question": "question", "image": "image", "category": "category",}
    },
    "ys-zong/VLGuard": {
        "categories": None,
        "field_map": {"question": "question", "image": "image", "category": "category",}
    },
    "Foreshhh/vlsbench": {
        "categories": None,
        "field_map": {"question": "instruction", "image": "image", "category": "category",}
    },
    "MMMU/MMMU_Pro": {
        "categories": None,
        "field_map": {
            "question": "question",
            "image": "image",
            "category": "category",
            "answer": "answer",
            "id": "id",
        },
    },
    "AI4Math/MathVista": {
        "categories": MathVista_categories,
        "field_map": {
            "question": "query",
            "image": "decoded_image",
            "category": "question_type",
            "answer": "answer",
            "id": "pid"
        }
    },
    "Lin-Chen/MMStar": {
        "categories": MMStar_categories,
        "field_map": {
            "question": "question",
            "image": "image",
            "category": "category",
            "answer": "answer",
            "id": "index"
        }
    },
    "lmms-lab-encoder/ai2d": {
        "categories": AI2D_categories,
        "field_map": {
            "question": "question",
            "image": "image",
            "category": "category",
            "answer": "answer"
        }
    }
}
DEFAULT_FIELD_MAP = {"question": "question", "image": "image", "category": "category"}


def load_eval_dataset(
    dataset_id: str,
    split: str = "evaluation",
    benchmark_name: Optional[str] = None,
    field_map: Optional[dict] = None,
    categories: Optional[list] = None,
    dedup_questions: bool = False,
) -> List[Sample]:
    """Convert any registered (or field_map-annotated) HF dataset into a
    list of Samples, the same way load_beavertails_v() does below.

    `field_map` overrides `DATASET_CONFIGS[dataset_id]["field_map"]` (or the
    generic {"question","image","category"} default) if the dataset uses
    different column names -- pass it directly for datasets you haven't
    added to DATASET_CONFIGS yet, e.g.:

        load_eval_dataset("some-org/SPA-VL", split="test",
                           field_map={"question": "prompt", "image": "image", "category": "category"})
    """
    cfg = DATASET_CONFIGS.get(dataset_id, {})
    field_map = field_map or cfg.get("field_map", DEFAULT_FIELD_MAP)
    benchmark_name = benchmark_name or dataset_id.split("/")[-1]

    dataset = get_dataset(dataset_id=dataset_id, split=split, categories=categories)

    q_field = field_map["question"]
    img_field = field_map.get("image")
    cat_field = field_map.get("category")
    ans_field = field_map.get("answer")
    id_field = field_map.get("id")

    samples = []
    seen = set()
    for i, row in enumerate(dataset):
        category = row.get(cat_field) if cat_field else None
        key = (category, row[q_field])
        if dedup_questions and key in seen:
            continue
        seen.add(key)
        # === MMMU-Pro: reuse the dataset's own id (e.g. "test_Accounting_1")
        # when field_map provides one, so the future scorer can match a
        # generation back to the official MMMU-Pro answer key. Falls back to
        # the original synthetic id for every other dataset.
        if id_field:
            sample_id = f"{benchmark_name.lower()}_{row[id_field]}"
        else:
            sample_id = f"{benchmark_name.lower()}_{category or 'na'}_{i:06d}"
        samples.append(
            Sample(
                id=sample_id,
                benchmark=benchmark_name,
                question=row[q_field],
                image=row.get(img_field) if img_field else None,
                category=category,
                answer=row.get(ans_field) if ans_field else None,  # === MMMU-Pro
            )
        )
    return samples


def load_beavertails_v(
    dataset_id: str = "saferlhf-v/BeaverTails-V",
    split: str = "evaluation",
    benchmark_name: str = "BeaverTails-V",
    dedup_questions: bool = True,
) -> List[Sample]:
    """Thin, backward-compatible wrapper around load_eval_dataset() for
    BeaverTails-V specifically.

    Fields per the dataset card: question (str), image (PIL.Image), category
    (str), image_severity (int 1-3), response (str), is_response_safe
    ("yes"/"no"). `response`/`is_response_safe` are pre-existing preference
    annotations from other VLMs -- we ignore them since our pipeline
    generates and judges its own policy responses.

    Each (question, image) pair appears twice in the raw data (once per
    paired preference response), so dedup_questions=True keeps only the
    first occurrence per unique question text within a category.
    """
    return load_eval_dataset(
        dataset_id, split=split, benchmark_name=benchmark_name, dedup_questions=dedup_questions
    )


def batched(samples: List[Sample], batch_size: int) -> Iterator[List[Sample]]:
    for i in range(0, len(samples), batch_size):
        yield samples[i : i + batch_size]


def index_by_id(samples: List[Sample]) -> dict:
    """id -> Sample lookup, used by judge-only scripts to re-attach the
    original image to a generation record that was saved to JSONL (and so
    doesn't carry the in-memory PIL image with it)."""
    return {s.id: s for s in samples}


def load_for_data_arg(data: str, split: str = "evaluation") -> List[Sample]:
    """Shared --data dispatch used by all CLI scripts: a .jsonl path is
    loaded via load_samples(), anything else is treated as a HF dataset id
    and loaded via load_eval_dataset() (which knows BeaverTails-V out of the
    box and can be pointed at other benchmarks via DATASET_CONFIGS /
    field_map)."""
    if data.endswith(".jsonl"):
        return load_samples(data)
    # === MMMU-Pro: only a "test" split exists on the Hub for this dataset;
    # auto-correct the default so `--data MMMU/MMMU_Pro` works without also
    # having to pass `--split test` explicitly.
    if data == "MMMU/MMMU_Pro" and split == "evaluation":
        split = "test"
    return load_eval_dataset(data, split=split)


def load_done_ids(out_path: str, id_field: str = "id") -> set:
    """Read back the ids already present in a (possibly partially-written)
    output JSONL, so a run can resume instead of regenerating from scratch.
    Corrupt/truncated trailing lines (e.g. from a killed process) are
    skipped rather than raising.
    """
    import json
    import os

    done = set()
    if not os.path.exists(out_path):
        return done
    with open(out_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if id_field in row:
                done.add(row[id_field])
    return done