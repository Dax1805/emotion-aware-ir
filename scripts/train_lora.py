"""""
This module is experimental and is not directly integrated into the IR system.
It serves as a placeholder for future incorporation of dynamic user feedback
into model fine-tuning workflows.
"""

import os
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DistilBertConfig,
    TrainingArguments,
    Trainer,
)
from peft import get_peft_model, LoraConfig, TaskType

# Define project root and paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
data_path = os.path.join(BASE_DIR, "data", "processed", "final_finetune_dataset.jsonl")
output_dir = os.path.join(BASE_DIR, "models", "lora_emotion_model")
os.makedirs(output_dir, exist_ok=True)

# Load dataset (in JSONL format)
dataset = load_dataset("json", data_files=data_path, split="train")

# Emotion label mapping to IDs
label_map = {
    "Angry": 0,
    "Hopeful": 1,
    "Fearful": 2,
    "Frustrated": 3,
    "Neutral": 4,
    "Empowered": 5
}

# Model and tokenizer name
model_name = "joeddav/distilbert-base-uncased-go-emotions-student"
tokenizer = AutoTokenizer.from_pretrained(model_name)


# Tokenize and encode emotion labels
def tokenize(example):
    label_str = example["label"]
    label_id = label_map.get(label_str, -1)

    tokenized = tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=256
    )
    tokenized["labels"] = int(label_id)
    return tokenized


# Apply preprocessing
dataset = dataset.map(tokenize, remove_columns=["text", "label"])
dataset = dataset.filter(lambda x: x["labels"] != -1)

# Configure LoRA (Low-Rank Adaptation)
peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_lin", "v_lin"]
)

# Load base model with modified config (6 emotion classes)
config = DistilBertConfig.from_pretrained(model_name, num_labels=6)
base_model = AutoModelForSequenceClassification.from_pretrained(
    model_name, config=config, ignore_mismatched_sizes=True
)

# Attach LoRA adapters to the base model
model = get_peft_model(base_model, peft_config)

# Define training parameters
training_args = TrainingArguments(
    output_dir=output_dir,
    label_names=["labels"],
    per_device_train_batch_size=32,
    num_train_epochs=20,
    learning_rate=1e-5,
    weight_decay=0.01,
    logging_dir=os.path.join(BASE_DIR, "logs"),
    logging_steps=10,
    save_strategy="epoch",
    evaluation_strategy="no",
    report_to="none"
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer
)

# Train the model
trainer.train()

# Save LoRA adapter
model.save_pretrained(output_dir)
print(f"Fine-tuned model with LoRA saved to: {output_dir}")

