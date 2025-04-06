"""""
This module is experimental and is not directly integrated into the IR system.
It serves as a placeholder for future incorporation of dynamic user feedback
into model fine-tuning workflows.
"""

import os
import json
import pandas as pd
from app.emotion_toggle import get_cont_emotions

# Define project paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
feedback_path = os.path.join(BASE_DIR, "data", "raw", "feedback_log.csv")
output_dir = os.path.join(BASE_DIR, "data", "processed")
output_path = os.path.join(output_dir, "finetune_from_feedback.jsonl")

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Load feedback CSV
df = pd.read_csv(feedback_path)

output = []

# Process each feedback entry
for _, row in df.iterrows():
    if row["feedback"] == "👎 Not Relevant":
        emotion = row["emotion"]
        mode = row["mode"]
        text = row["doc_content"]

        # Determine label based on mode
        if mode == "aligned":
            label = emotion  # we expected this, but it failed
        elif mode == "cont":
            cont_emotions = get_cont_emotions(emotion)
            if cont_emotions:
                label = cont_emotions[0]  # take the first contrasting label
            else:
                continue  # skip if no alternative found
        else:
            continue  # unknown mode, skip

        output.append({
            "text": text,
            "label": label
        })

# Save to JSONL for fine-tuning
with open(output_path, "w", encoding="utf-8") as f:
    for ex in output:
        f.write(json.dumps(ex) + "\n")

print(f" Saved {len(output)} training samples to {output_path}")