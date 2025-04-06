from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
from typing import List

# Base GoEmotions model (fine-tuned DistilBERT)
model_name = "joeddav/distilbert-base-uncased-go-emotions-student"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Full GoEmotions label set (27 emotions + Neutral)
goemotions_labels = [
    "admiration", "amusement", "anger", "annoyance", "approval",
    "caring", "confusion", "curiosity", "desire", "disappointment",
    "disapproval", "disgust", "embarrassment", "excitement", "fear",
    "gratitude", "grief", "joy", "love", "nervousness", "neutral",
    "optimism", "pride", "realization", "relief", "remorse", "sadness", "surprise"
]

# Map GoEmotions labels to simplified emotion categories
goemotion_to_custom = {
    "optimism": "Hopeful",
    "anticipation": "Hopeful",
    "joy": "Hopeful",
    "anger": "Angry",
    "annoyance": "Angry",
    "fear": "Fearful",
    "disappointment": "Frustrated",
    "pride": "Empowered",
    "gratitude": "Empowered",
    "neutral": "Neutral"
}

# Custom emotion thresholds for classification confidence
custom_thresholds = {
    "empowered": 0.72,
    "hopeful": 0.35,
    "neutral": 0.5
}


def predict_emotions(text: str, threshold: float = 0.5) -> List[str]:
    """
    Predict simplified emotion labels from input text using GoEmotions model.

    Args:
        text (str): Input sentence or phrase.
        threshold (float): Default classification threshold (used if no custom threshold exists).

    Returns:
        List[str]: List of simplified emotion categories.
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True)

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = F.sigmoid(logits)[0]  # Multi-label probabilities

    detected = []

    for i, p in enumerate(probs):
        label = goemotions_labels[i]
        local_thresh = custom_thresholds.get(label.lower(), threshold)
        if p > local_thresh:
            detected.append(label)

    # Convert GoEmotions to simplified emotion labels
    mapped = {goemotion_to_custom[emo] for emo in detected if emo in goemotion_to_custom}

    # Fallback to Neutral if nothing is confidently predicted
    return list(mapped) if mapped else ["Neutral"]
