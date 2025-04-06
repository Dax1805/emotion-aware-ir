import pandas as pd
import os
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Define path
feedback_path = os.path.join("data", "raw", "feedback_log.csv")

# Load feedback data
if not os.path.exists(feedback_path):
    raise FileNotFoundError(f"No feedback_log.csv found at: {feedback_path}")

df = pd.read_csv(feedback_path)

# Normalize values
df["feedback"] = df["feedback"].map({
    "👍 Relevant": 1,
    "👎 Not Relevant": 0
})

# Drop any missing or malformed feedback
df = df.dropna(subset=["query", "feedback", "doc_title"])

# Assume relevance ground truth = feedback
y_true = df["feedback"]
y_pred = [1] * len(df)  # Your system retrieved all of these → assume all predicted as relevant

# Basic metrics
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
acc = accuracy_score(y_true, y_pred)

print(f"✅ IR Evaluation from Feedback")
print(f"Precision:  {precision:.2f}")
print(f"Recall:     {recall:.2f}")
print(f"F1-score:   {f1:.2f}")
print(f"Accuracy:   {acc:.2f}")

# Confusion matrix
cm = confusion_matrix(y_true, y_pred, labels=[1, 0])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Relevant", "Not Relevant"])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix (User Feedback)")
plt.show()
