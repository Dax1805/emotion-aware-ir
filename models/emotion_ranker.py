from typing import List, Dict
from models.emotion_classifier import predict_emotions
from app.emotion_toggle import get_cont_emotions


def fuzzy_emotion_rerank(
        results: List[Dict],
        user_emotion: str,
        mode: str = "aligned",
        alpha: float = 0.7
) -> List[Dict]:
    """
    Apply emotion-aware reranking to search results using a fuzzy logic blend
    of semantic relevance and emotional alignment.

    Args:
        results (List[Dict]): List of results, each with 'text' and 'score'.
        user_emotion (str): Emotion selected by the user.
        mode (str): Either 'aligned' (reinforce same emotion) or 'cont' (contrast).
        alpha (float): Weight for semantic relevance (0.0–1.0). 1-alpha is used for emotion.

    Returns:
        List[Dict]: Reranked results, each including emotion and final score.
    """
    target_emotions = (
        set(get_cont_emotions(user_emotion)) if mode == "cont" else {user_emotion}
    )

    reranked = []

    for item in results:
        if not isinstance(item, dict) or "text" not in item or "score" not in item:
            continue

        text = item["text"]
        relevance_score = item["score"]

        # Predict emotions from the text
        predicted_emotions = predict_emotions(text)

        # Determine emotion match (1.0 if any target emotion found)
        emotion_score = 1.0 if any(e in predicted_emotions for e in target_emotions) else 0.0

        # Weighted combination of relevance and emotion
        final_score = (alpha * relevance_score) + ((1 - alpha) * emotion_score)

        reranked.append({
            "text": text,
            "original_score": relevance_score,
            "emotion_score": emotion_score,
            "final_score": final_score
        })

    # Sort results by final combined score
    reranked.sort(key=lambda x: x["final_score"], reverse=True)
    return reranked
