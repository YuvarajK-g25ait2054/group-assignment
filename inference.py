"""
Inference script for sentiment classification
Loads model from HuggingFace Hub and runs predictions
"""
import os
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

# Get model name from environment variable or use default
MODEL_NAME = os.getenv("MODEL_NAME", "YuvarajK-g25ait2054/distilbert-imdb-sentiment")

print(f"Loading model: {MODEL_NAME}")
# Load tokenizer and model explicitly to avoid token_type_ids issue with DistilBERT
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
# DistilBERT doesn't use token_type_ids, so we need to remove it from tokenizer outputs
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer, 
                     return_token_type_ids=False)
print("Model loaded successfully!")

# Test predictions
test_texts = [
    "This movie was absolutely fantastic!",
    "Terrible film, waste of time.",
    "Pretty good, I enjoyed it."
]

print("\n=== Running Inference ===")
for text in test_texts:
    result = classifier(text)[0]
    print(f"\nInput: {text}")
    print(f"Prediction: {result['label']} (confidence: {result['score']:.4f})")
