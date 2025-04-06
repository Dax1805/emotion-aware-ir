import pandas as pd
from sentence_transformers import SentenceTransformer, util
from models.emotion_ranker import fuzzy_emotion_rerank

# Load pretrained Sentence-BERT model
sbert_model = SentenceTransformer("all-MiniLM-L6-v2")


def load_corpus(path: str = "data/corpus/corpus.csv") -> pd.DataFrame:
    """
    Load the document corpus from a CSV file.

    Args:
        path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded DataFrame containing at least a 'title' column.
    """
    return pd.read_csv(path)


def rank_documents_by_query(
        corpus_df: pd.DataFrame,
        query: str,
        emotion_filter: str,
        mode: str = "aligned",
        alpha: float = 0.7,
        beta: float = 0.3
) -> pd.DataFrame:
    """
    Rank documents based on query similarity and emotional alignment or contrast.

    Args:
        corpus_df (pd.DataFrame): DataFrame containing at least a 'title' column.
        query (str): Search query from the user.
        emotion_filter (str): User emotion or contrast emotion to match against.
        mode (str): 'aligned' or 'cont' for contrast mode.
        alpha (float): Weight for semantic similarity (0 to 1).
        beta (float): Weight for emotional match score (0 to 1).

    Returns:
        pd.DataFrame: Ranked DataFrame with added columns:
            - 'relevance': semantic similarity score
            - 'emotion_match': emotion alignment score
            - 'final_score': combined ranking score
    """
    # Extract text content
    doc_texts = corpus_df["title"].tolist()

    # Compute SBERT embeddings
    doc_embeddings = sbert_model.encode(doc_texts, convert_to_tensor=True)
    query_embedding = sbert_model.encode(query, convert_to_tensor=True)

    # Calculate cosine similarity between query and documents
    cosine_scores = util.cos_sim(query_embedding, doc_embeddings)[0].cpu().numpy()

    # Compute emotion match scores
    emotion_confidences = []
    for idx, text in enumerate(doc_texts):
        relevance_score = cosine_scores[idx]
        results = [{"text": text, "score": relevance_score}]
        reranked = fuzzy_emotion_rerank(results, emotion_filter, mode=mode)
        emotion_score = reranked[0]["emotion_score"] if reranked else 0.0
        emotion_confidences.append(emotion_score)

    # Combine scores using weighted sum
    final_scores = [
        (alpha * rel) + (beta * emo)
        for rel, emo in zip(cosine_scores, emotion_confidences)
    ]

    # Add scoring columns to the corpus
    corpus_df["relevance"] = cosine_scores
    corpus_df["emotion_match"] = emotion_confidences
    corpus_df["final_score"] = final_scores

    # Return ranked results
    return corpus_df.sort_values(by="final_score", ascending=False)
