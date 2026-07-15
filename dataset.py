from datasets import load_dataset, concatenate_datasets

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
    "insulting_and_harassing_behavior"
]

def get_dataset(dataset_id="saferlhf-v/BeaverTails-V", split="train", test_size=0.1, seed=42):

    if dataset_id == "saferlhf-v/BeaverTails-V" or dataset_id == "PKU-Alignment/BeaverTails-V":
        print(f"[DEBUG] Loading BeaverTails-V from dataset_id {dataset_id}")
        print(f"[DEBUG] Loading {split}")
        
        datasets = [load_dataset(dataset_id, c)[split] for c in HARM_CATEGORIES]
        dataset = concatenate_datasets(datasets)

        if split == "train":
            print(f"[DEBUG] Splitting the dataset {(1-test_size)*100}-to-{test_size*100} train-to-val ratio with seed {seed}")
            dataset = dataset.train_test_split(test_size=test_size, seed=seed)
        elif split == "evaluation":
            pass 
        else:
            raise KeyError(f"[ERROR] Invalid split '{split}' for {dataset_id}. Valid splits are: ['train', 'evaluation']")
    return dataset