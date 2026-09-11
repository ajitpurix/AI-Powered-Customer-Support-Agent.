import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, classification_report

TRAIN_PATH = Path("data/golden/train.csv")
GOLDEN_PATH = Path("data/golden/golden_set.csv")

train_df = pd.read_csv(TRAIN_PATH)
golden_df = pd.read_csv(GOLDEN_PATH)

y_train = train_df["human_intent"].astype(str)
y_test = golden_df["human_intent"].astype(str)

# Most frequent training intent
majority_intent = y_train.value_counts().idxmax()

# Predict the same intent for every example
predictions = [majority_intent] * len(y_test)

accuracy = accuracy_score(y_test, predictions)
macro_f1 = f1_score(
    y_test,
    predictions,
    average="macro",
    zero_division=0
)
weighted_f1 = f1_score(
    y_test,
    predictions,
    average="weighted",
    zero_division=0
)

print("=" * 70)
print("MAJORITY BASELINE")
print("=" * 70)

print(f"Most common intent: {majority_intent}")
print(f"Accuracy:           {accuracy:.4f}")
print(f"Macro F1:           {macro_f1:.4f}")
print(f"Weighted F1:        {weighted_f1:.4f}")

print("\n===== CLASSIFICATION REPORT =====")

print(
    classification_report(
        y_test,
        predictions,
        zero_division=0
    )
)

# Save predictions
results = golden_df[
    ["customer_tweet_id", "customer_text", "human_intent"]
].copy()

results["predicted_intent"] = predictions

output_path = Path("data/golden/majority_predictions.csv")

results.to_csv(output_path, index=False)

print(f"\nPredictions saved to: {output_path}")