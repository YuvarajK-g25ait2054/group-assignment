"""
Data Preparation & Normalisation
Dataset: IMDB Movie Reviews (Sentiment Classification)
Labels: 0 = negative, 1 = positive
"""

import json
import re
from datasets import load_dataset


def clean_text(text):
    text = re.sub(r'<.*?>', '', text)       # remove HTML tags like <br />
    text = re.sub(r'\s+', ' ', text).strip() # remove extra spaces/newlines
    return text


def main():
    print("Loading IMDB dataset...")
    dataset = load_dataset("imdb")

    # --- Inspect raw data ---
    print("\n--- Raw Data Inspection ---")
    print(f"Train size : {len(dataset['train'])}")
    print(f"Test size  : {len(dataset['test'])}")
    print(f"Features   : {dataset['train'].features}")
    print(f"Sample text: {dataset['train'][0]['text'][:200]}")
    print(f"Sample label: {dataset['train'][0]['label']}")

    # --- Class distribution ---
    labels_list = dataset["train"]["label"]
    print(f"\nClass distribution (train):")
    print(f"  Negative (0): {labels_list.count(0)}")
    print(f"  Positive (1): {labels_list.count(1)}")

    # --- Clean the data ---
    print("\nCleaning text (removing HTML tags, extra whitespace)...")
    dataset = dataset.map(lambda x: {"text": clean_text(x["text"])})
    print(f"Sample after cleaning: {dataset['train'][0]['text'][:200]}")

    # --- Create label mapping ---
    label_names = dataset["train"].features["label"].names
    id2label = {i: label for i, label in enumerate(label_names)}
    label2id = {label: i for i, label in id2label.items()}

    print(f"\nLabel mapping: {id2label}")

    # --- Save id2label.json ---
    with open("id2label.json", "w") as f:
        json.dump(id2label, f, indent=2)
    print("\nSaved id2label.json")

    # --- Save a small sample for local testing (optional) ---
    small_train = dataset["train"].select(range(100))
    small_train.to_json("sample_data.json")
    print("Saved sample_data.json (100 rows for local testing)")

    print("\nDone! Commit only id2label.json to GitHub (not the full dataset).")


if __name__ == "__main__":
    main()
