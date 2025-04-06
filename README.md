# Emotion-Aware Information Retrieval

This project is an emotion-aware search engine that retrieves and reranks real Reddit posts based on how the user is feeling — enabling more empathetic, contrastive, and emotionally aligned exploration of social justice content.

---

## ✨ Features

- 🔍 Real Reddit post retrieval from [`r/politics`](https://www.reddit.com/r/politics)
- 🤖 Emotion classification using a fine-tuned GoEmotions-based model
- 💡 Emotion-aligned and contrastive reranking modes
- 🧠 SBERT-powered semantic similarity scoring
- 🌛 Interactive Streamlit interface with:
    - Emotion toggles
    - Alpha slider to control emotion vs. semantic balance
    - Pagination to improve performance
    - Binary feedback collection (👍 / 👎)
- ♻️ Feedback integration pipeline for future model fine-tuning

---

## 🌝 Emotion Categories

| Emotion      | Description                                  |
|--------------|----------------------------------------------|
| Hopeful      | Joy, optimism, anticipation                  |
| Angry        | Anger, annoyance                             |
| Frustrated   | Disappointment, helplessness                 |
| Fearful      | Anxiety, threat perception                   |
| Empowered    | Pride, gratitude, approval                   |
| Neutral      | Objective, emotionally flat content          |

---

## ⚙️ Setup Instructions

```bash
git clone https://github.com/yourusername/emotion-ir
cd emotion-ir
pip install -r requirements.txt
```

---

## 🚀 How to Run

To launch the interactive retrieval system, run:

```bash
streamlit run app/run_app.py
```

> ⚠️ Make sure your corpus exists at `data/corpus/corpus.csv` before launching the app. You can populate it using:
```bash
python scripts/pull_reddit.py
```

This fetches the latest Reddit posts and updates the corpus, avoiding duplicates.

---

## 📈 Model Training and Fine-Tuning

This project includes:

- A script for LoRA-based fine-tuning (`scripts/finetune_lora.py`)
- Dataset generation from misclassified samples + user feedback (`scripts/merge_finetune_dataset.py`)
- Evaluation of the emotion classifier (`scripts/evaluate_classifier.py`)
- Future-ready IR evaluation using feedback logs

---

## 📁 Project Structure

```
emotion-ir/
|
├── app/                       # Streamlit UI + toggles
├── data/                      # Datasets, logs, and corpus
├── models/                    # Classifier, reranker, retriever
├── scripts/                   # Data collection and training scripts
├── logs/                      # Training logs
└── run_app.py                 # Main Streamlit entrypoint
```

---

## 🧪 Evaluation

Emotion classification currently achieves ~61% accuracy on our curated custom emotion dataset. We include a feedback-to-finetune pipeline and scripts for computing IR metrics like Precision@k and MRR based on user annotations.

---

## 📌 Notes

- This is a prototype and not deployed live.
- All data collected from Reddit is for research purposes only.
- The interface and classifier are extensible to other domains and emotion categories.

---

## 📄 License

MIT License. Feel free to fork, adapt, or build on this work for educational or non-commercial use.

