from typing import List, Dict, Union
from sklearn.metrics.pairwise import cosine_similarity
from models.emotion_classifier import predict_emotions


def rank_documents(
        query: str,
        documents: List[Dict[str, Union[str, any]]],
        user_emotion: Union[str, None] = None,
        contrast_mode: bool = False
) -> List[Dict]:
    """
    Rank a list of documents based on semantic relevance to the query and, optionally,
    emotional alignment or contrast with the user's emotion.

    Args:
        query (str): The user's search query.
        documents (List[Dict]): Each document must include:
            - 'text': textual content
            - 'embedding': precomputed vector (NumPy or Torch)
        user_emotion (str, optional): Emotion selected by the user. Default is None.
        contrast_mode (bool): If True, boosts documents that contrast with the emotion.

    Returns:
        List[Dict]: Ranked list of documents with similarity and optional emotion scores.
    """
    if not documents:
        return []

    # Derive embedding class and prepare query vector
    query_embedding = documents[0]['embedding'].__class__(query)
    if hasattr(query_embedding, 'detach'):
        query_embedding = query_embedding.detach().cpu().numpy()

    # Calculate cosine similarity to each document
    similarities = [
        cosine_similarity(query_embedding.reshape(1, -1), doc['embedding'].reshape(1, -1))[0][0]
        for doc in documents
    ]

    # Assign base similarity scores
    for i, doc in enumerate(documents):
        doc['similarity'] = similarities[i]

    if user_emotion:
        for doc in documents:
            predicted_emotions = predict_emotions(doc['text'])
            doc['predicted_emotions'] = predicted_emotions

            if contrast_mode:
                # Reward contrast, penalize alignment
                doc['emotion_score'] = 1.0 if user_emotion not in predicted_emotions else 0.0
            else:
                # Reward alignment
                doc['emotion_score'] = 1.0 if user_emotion in predicted_emotions else 0.0

        # Combine scores: 80% relevance + 20% emotion match
        for doc in documents:
            doc['final_score'] = 0.8 * doc['similarity'] + 0.2 * doc['emotion_score']

        documents.sort(key=lambda x: x['final_score'], reverse=True)
    else:
        # Default: sort by relevance only
        documents.sort(key=lambda x: x['similarity'], reverse=True)

    return documents
