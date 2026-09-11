import pandas as pd
from pathlib import Path

CUSTOMER_PATH = Path("data/processed/apple_customer_messages.csv")
REPLY_PATH = Path("data/processed/apple_support_replies.csv")

OUTPUT_PATH = Path("data/processed/apple_conversations.csv")


print("Loading AppleSupport data...")

customers = pd.read_csv(CUSTOMER_PATH, dtype=str)
replies = pd.read_csv(REPLY_PATH, dtype=str)

print(f"Customer messages : {len(customers):,}")
print(f"Support replies   : {len(replies):,}")


# Rename columns so the merge is easy to understand
replies = replies.rename(
    columns={
        "tweet_id": "reply_tweet_id",
        "author_id": "reply_author_id",
        "created_at": "reply_created_at",
        "text": "reply_text",
    }
)

# A customer's response_tweet_id points to the support tweet
# that replied to that customer.
conversations = customers.merge(
    replies[
        [
            "reply_tweet_id",
            "reply_created_at",
            "reply_text",
        ]
    ],
    left_on="response_tweet_id",
    right_on="reply_tweet_id",
    how="inner",
)

# Rename customer fields
conversations = conversations.rename(
    columns={
        "tweet_id": "customer_tweet_id",
        "author_id": "customer_id",
        "created_at": "customer_created_at",
        "text": "customer_text",
    }
)

# Keep only useful fields
conversations = conversations[
    [
        "customer_tweet_id",
        "customer_id",
        "customer_created_at",
        "customer_text",
        "reply_tweet_id",
        "reply_created_at",
        "reply_text",
    ]
]

# Remove empty messages
conversations = conversations.dropna(
    subset=["customer_text", "reply_text"]
)

# Remove duplicate pairs
conversations = conversations.drop_duplicates(
    subset=["customer_tweet_id", "reply_tweet_id"]
)

# Save
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

conversations.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\n===== RESULT =====")
print(f"Conversation pairs: {len(conversations):,}")

print("\n===== EXAMPLE CONVERSATIONS =====")

for _, row in conversations.head(5).iterrows():

    print("\nCUSTOMER:")
    print(row["customer_text"])

    print("\nAPPLESUPPORT:")
    print(row["reply_text"])

    print("-" * 80)

print(f"\nSaved to: {OUTPUT_PATH}")