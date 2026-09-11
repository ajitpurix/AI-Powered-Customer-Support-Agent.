import pandas as pd
from pathlib import Path

DATA_PATH = Path("data/raw/twcs.csv")

print("Loading dataset...")
df = pd.read_csv(DATA_PATH)

print(f"Total tweets: {len(df):,}")

# Separate customer and support tweets
customer_tweets = df[df["inbound"] == True]
support_tweets = df[df["inbound"] == False]

print(f"Customer tweets: {len(customer_tweets):,}")
print(f"Support tweets: {len(support_tweets):,}")

# Count tweets written by each author
author_counts = df["author_id"].value_counts()

print("\n===== TOP AUTHORS =====")
print(author_counts.head(30))

# Support accounts tend to have many outbound tweets.
support_author_counts = support_tweets["author_id"].value_counts()

print("\n===== TOP SUPPORT ACCOUNTS =====")
print(support_author_counts.head(30))