import pandas as pd
from pathlib import Path

INPUT_PATH = Path("data/processed/apple_conversations.csv")
OUTPUT_PATH = Path("data/golden/labeling_candidates.csv")

print("Loading AppleSupport conversations...")

df = pd.read_csv(INPUT_PATH)

df = df.dropna(subset=["customer_text", "reply_text"])

# Remove extremely short messages
df["text_length"] = df["customer_text"].astype(str).str.len()

df = df[df["text_length"] >= 20]

# Reproducible random sampling
candidates = df.sample(
    n=min(300, len(df)),
    random_state=42
)

# Keep only useful columns
candidates = candidates[
    [
        "customer_tweet_id",
        "customer_text",
        "reply_text"
    ]
].copy()

# Empty intent column for human labeling
candidates["intent"] = ""

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

candidates.to_csv(
    OUTPUT_PATH,
    index=False
)

print()
print("===== LABELING CANDIDATES =====")
print(f"Candidates: {len(candidates)}")
print(f"Saved to: {OUTPUT_PATH}")