import pandas as pd
from pathlib import Path

INPUT_PATH = Path("data/processed/apple_conversations.csv")
OUTPUT_PATH = Path("data/golden/intent_labeling.csv")

# Number of examples to manually label
N = 500

print("Loading conversations...")

df = pd.read_csv(INPUT_PATH)

# Remove empty messages
df = df.dropna(subset=["customer_text"])

# Clean very short messages
df["customer_text"] = df["customer_text"].astype(str)

df = df[df["customer_text"].str.len() >= 15]

# Random sample
sample = df.sample(
    n=min(N, len(df)),
    random_state=42
).copy()

# Add empty label
sample["intent"] = ""

# Keep useful columns
sample = sample[
    [
        "customer_tweet_id",
        "customer_text",
        "reply_text",
        "intent"
    ]
]

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

sample.to_csv(
    OUTPUT_PATH,
    index=False
)

print(f"\nCreated {len(sample)} examples.")
print(f"File: {OUTPUT_PATH}")
print("\nOpen this file and fill the 'intent' column manually.")