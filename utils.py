"""
Load Model and Tokenizer from HuggingFace
Model: distilbert-base-uncased
"""

import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def load_model_and_tokenizer(model_name="distilbert-base-uncased", id2label_path="id2label.json"):
    """Load tokenizer and model with label mappings."""
    
    # Load label mappings
    with open(id2label_path, "r") as f:
        id2label = json.load(f)
    
    # Convert keys to integers
    id2label = {int(k): v for k, v in id2label.items()}
    label2id = {v: k for k, v in id2label.items()}
    
    print(f"Loading model: {model_name}")
    print(f"Number of labels: {len(id2label)}")
    print(f"Label mapping: {id2label}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load model with correct number of labels
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(id2label),
        id2label=id2label,
        label2id=label2id
    )
    
    print(f"Model loaded successfully!")
    print(f"Model parameters: {model.num_parameters():,}")
    
    return tokenizer, model


if __name__ == "__main__":
    # Test loading
    tokenizer, model = load_model_and_tokenizer()
    
    # Test tokenization
    test_text = "This movie was absolutely fantastic!"
    inputs = tokenizer(test_text, return_tensors="pt")
    print(f"\nTest tokenization:")
    print(f"Input text: {test_text}")
    print(f"Tokenized IDs: {inputs['input_ids'][0][:10]}...")
    print(f"Tokens: {tokenizer.convert_ids_to_tokens(inputs['input_ids'][0][:10])}")
