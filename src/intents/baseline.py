import pandas as pd
from pathlib import Path

DATA_PATH = Path("data/golden/intent_labeling.csv")

df = pd.read_csv(DATA_PATH)

# Remove rows without labels
df = df.dropna(subset=["intent"])
df = df[df["intent"].astype(str).str.strip() != ""]

print(f"Labelled examples: {len(df)}")

# Find most common intent
majority_intent = df["intent"].value_counts().idxmax()

print("\n===== CLASS DISTRIBUTION =====")
print(df["intent"].value_counts())

print("\n===== MAJORITY BASELINE =====")
print(f"Most common intent: {majority_intent}")
print(f"Baseline accuracy: {df['intent'].eq(majority_intent).mean():.4f}")