import praw
import pandas as pd
import os

# Reddit API credentials
reddit = praw.Reddit(
    client_id="w82Lh2HLCUEf2ERBWzga8Q",
    client_secret="gtKi-qm3qc-l5pDZtfxfLjAEnwmDVQ",
    username="TableIllustrious9174",
    password="India!23",
    user_agent="emotion-ir-script"
)


def fetch_titles(subreddit_name: str = "politics", limit: int = 1000) -> pd.DataFrame:
    """
    Fetches recent post titles from a specified subreddit.

    Args:
        subreddit_name (str): Name of the subreddit to fetch posts from.
        limit (int): Number of posts to fetch.

    Returns:
        pd.DataFrame: DataFrame with columns: id, title, score, created_utc, url
    """
    titles = []
    for submission in reddit.subreddit(subreddit_name).new(limit=limit):
        titles.append({
            "id": submission.id,
            "title": submission.title,
            "score": submission.score,
            "created_utc": submission.created_utc,
            "url": submission.url
        })
    return pd.DataFrame(titles)


# Output directory
os.makedirs("data/corpus", exist_ok=True)
corpus_path = "data/corpus/corpus.csv"

# Load existing data if available
if os.path.exists(corpus_path):
    existing_df = pd.read_csv(corpus_path)
    seen_ids = set(existing_df["id"])
else:
    existing_df = pd.DataFrame()
    seen_ids = set()

# Fetch and filter out already seen posts
new_df = fetch_titles()
new_df = new_df[~new_df["id"].isin(seen_ids)]

# Combine and save the updated corpus
final_df = pd.concat([existing_df, new_df], ignore_index=True)
final_df.to_csv(corpus_path, index=False)

print(f"Fetched {len(new_df)} new posts.")
print(f"Total posts in corpus: {len(final_df)}.")
