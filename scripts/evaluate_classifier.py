import os
import sys
import pandas as pd
from collections import defaultdict, Counter
from sklearn.metrics import classification_report

# Local imports
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)
from models.emotion_classifier import predict_emotions

# Path to emotion-labeled dataset
DATA_PATH = os.path.join(BASE_DIR, "data", "custom_emotion_dataset")

# List of labeled sample files to evaluate
emotion_files = [
    "hopeful_samples.csv",
    "angry_samples.csv",
    "frustrated_samples.csv",
    "fearful_samples.csv",
    "empowered_samples.csv",
    "neutral_samples.csv"
]

# Storage for evaluation results
true_labels = []
pred_labels = []
misclassified = defaultdict(list)
distribution = Counter()

# Evaluate predictions for each file
for file in emotion_files:
    file_path = os.path.join(DATA_PATH, file)
    df = pd.read_csv(file_path)

    for _, row in df.iterrows():
        text = row["text"]
        true = row["primary_emotion"]
        predicted = predict_emotions(text)

        true_labels.append(true)

        if true in predicted:
            pred_labels.append(true)
        else:
            pred_labels.append(predicted[0] if predicted else "Unclassified")
            misclassified[true].append((text, predicted))

        for p in predicted:
            distribution[p] += 1
        if not predicted:
            distribution["Unclassified"] += 1

# Display evaluation metrics
print("\nClassification Report (based on primary_emotion match):\n")
print(classification_report(true_labels, pred_labels))

print("\nPrediction Distribution:\n")
print(distribution)

print("\nMisclassified Samples (showing 3 per emotion):\n")
for emotion, items in misclassified.items():
    print(f"\n{emotion} misclassified:")
    for text, pred in items[:3]:
        print(f"  → Text: {text}\n    Predicted: {pred}\n")
