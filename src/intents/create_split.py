import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

INPUT_PATH = Path("data/golden/labeling_candidates.csv")

TRAIN_PATH = Path("data/golden/train.csv")
GOLDEN_PATH = Path("data/golden/golden_set.csv")

df = pd.read_csv(INPUT_PATH)

# Make sure labels exist
df = df.dropna(subset=["human_intent"])
df["human_intent"] = df["human_intent"].astype(str).str.strip()

# Remove accidental empty labels
df = df[df["human_intent"] != ""]

print(f"Total labelled examples: {len(df)}")

# Stratified split
train_df, golden_df = train_test_split(
    df,
    test_size=150,
    random_state=42,
    stratify=df["human_intent"]
)

# Keep only useful columns
columns = [
    "customer_tweet_id",
    "customer_text",
    "reply_text",
    "human_intent"
]

train_df = train_df[columns].copy()
golden_df = golden_df[columns].copy()

TRAIN_PATH.parent.mkdir(parents=True, exist_ok=True)

train_df.to_csv(TRAIN_PATH, index=False)
golden_df.to_csv(GOLDEN_PATH, index=False)

print("\n===== SPLIT CREATED =====")
print(f"Training examples : {len(train_df)}")
print(f"Golden examples   : {len(golden_df)}")

print("\n===== TRAIN DISTRIBUTION =====")
print(train_df["human_intent"].value_counts())

print("\n===== GOLDEN DISTRIBUTION =====")
print(golden_df["human_intent"].value_counts())

print("\nSaved:")
print(TRAIN_PATH)
print(GOLDEN_PATH)