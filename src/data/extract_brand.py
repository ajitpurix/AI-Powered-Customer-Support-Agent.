import pandas as pd
from pathlib import Path

DATA_PATH = Path("data/raw/twcs.csv")
OUTPUT_PATH = Path("data/processed/apple_support.csv")

BRAND = "AppleSupport"

print("Loading dataset...")

df = pd.read_csv(
    DATA_PATH,
    dtype={
        "tweet_id": str,
        "author_id": str,
        "response_tweet_id": str,
        "in_response_to_tweet_id": str,
    }
)

print(f"Total tweets: {len(df):,}")

# Select AppleSupport tweets
brand_tweets = df[df["author_id"] == BRAND].copy()

print(f"AppleSupport tweets: {len(brand_tweets):,}")

# Customer tweets that AppleSupport responded to
brand_tweet_ids = set(brand_tweets["tweet_id"])

customer_tweets = df[
    (df["inbound"] == True) &
    (df["response_tweet_id"].notna())
].copy()

# Keep customer tweets whose response points to AppleSupport
customer_tweets = customer_tweets[
    customer_tweets["response_tweet_id"].apply(
        lambda x: str(x) in brand_tweet_ids
    )
]

print(f"Customer messages with AppleSupport replies: {len(customer_tweets):,}")

# Keep useful columns
customer_tweets = customer_tweets[
    [
        "tweet_id",
        "author_id",
        "created_at",
        "text",
        "response_tweet_id",
        "in_response_to_tweet_id",
    ]
]

brand_tweets = brand_tweets[
    [
        "tweet_id",
        "author_id",
        "created_at",
        "text",
        "response_tweet_id",
        "in_response_to_tweet_id",
    ]
]

# Save both datasets
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

customer_tweets.to_csv(
    "data/processed/apple_customer_messages.csv",
    index=False
)

brand_tweets.to_csv(
    "data/processed/apple_support_replies.csv",
    index=False
)

print("\nSaved:")
print("data/processed/apple_customer_messages.csv")
print("data/processed/apple_support_replies.csv")