"""""
This module is experimental and is not directly integrated into the IR system.
It serves as a placeholder for future incorporation of dynamic user feedback
into model fine-tuning workflows.
"""
import os
import json
import pandas as pd

# Define project base directory
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Define paths to input and output files
misclassified_path = os.path.join(BASE_DIR, "data", "outputs", "eval_misclassifications.csv")
feedback_path = os.path.join(BASE_DIR, "data", "processed", "finetune_from_feedback.jsonl")
output_path = os.path.join(BASE_DIR, "data", "processed", "final_finetune_dataset.jsonl")

# Load misclassified samples (if available)
misclassified_data = []
if os.path.exists(misclassified_path) and os.path.getsize(misclassified_path) > 0:
    try:
        misclassified_df = pd.read_csv(misclassified_path)
        if {"text", "true"}.issubset(misclassified_df.columns):
            misclassified_data = [
                {"text": row["text"], "label": row["true"]}
                for _, row in misclassified_df.iterrows()
            ]
        else:
            print("Warning: 'text' and 'true' columns not found in misclassification file.")
    except Exception as e:
        print(f"Error reading misclassified CSV: {e}")

# Load feedback samples (if available)
feedback_data = []
if os.path.exists(feedback_path) and os.path.getsize(feedback_path) > 0:
    with open(feedback_path, "r", encoding="utf-8") as f:
        feedback_data = [json.loads(line.strip()) for line in f]

# Merge and deduplicate by text
combined_data = {entry["text"]: entry for entry in (misclassified_data + feedback_data)}

# Write to JSONL if data exists
if combined_data:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for example in combined_data.values():
            f.write(json.dumps(example) + "\n")
    print(f"Saved merged fine-tuning dataset with {len(combined_data)} samples to {output_path}")
else:
    print("No training samples found. Please generate evaluation or feedback data first.")
