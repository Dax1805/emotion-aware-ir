# Mapping of each emotion to a list of contrasting emotions
CONTRAST_EMOTION_MAP = {
    "Hopeful": ["Angry", "Frustrated"],
    "Angry": ["Hopeful", "Empowered"],
    "Fearful": ["Hopeful", "Empowered"],
    "Frustrated": ["Empowered", "Neutral"],
    "Empowered": ["Fearful", "Angry"],
    "Neutral": ["Hopeful", "Angry", "Fearful", "Empowered", "Frustrated"]
}

def get_cont_emotions(user_emotion):
    """
    Retrieve a list of contrasting emotions for a given emotion.

    Args:
        user_emotion (str): The selected user emotion.

    Returns:
        list[str]: List of contrasting emotions, or empty list if none found.
    """
    return CONTRAST_EMOTION_MAP.get(user_emotion, [])
