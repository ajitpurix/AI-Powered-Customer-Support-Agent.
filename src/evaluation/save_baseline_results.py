import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


TRAIN_PATH = Path("data/golden/train.csv")
GOLDEN_PATH = Path("data/golden/golden_set.csv")

MAJORITY_PATH = Path("data/golden/majority_predictions.csv")
TFIDF_PATH = Path("data/golden/tfidf_predictions.csv")

OUTPUT_PATH = Path("reports/baseline_results.json")


def evaluate(prediction_file):
    df = pd.read_csv(prediction_file)

    y_true = df["human_intent"]
    y_pred = df["predicted_intent"]

    return {
        "accuracy": round(
            accuracy_score(y_true, y_pred), 4
        ),
        "macro_f1": round(
            f1_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            ),
            4,
        ),
        "weighted_f1": round(
            f1_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0,
            ),
            4,
        ),
    }


results = {
    "majority_baseline": evaluate(MAJORITY_PATH),
    "tfidf_logistic_regression": evaluate(TFIDF_PATH),
}

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

with open(OUTPUT_PATH, "w") as f:
    json.dump(results, f, indent=2)

print("=" * 60)
print("BASELINE RESULTS")
print("=" * 60)

for model, metrics in results.items():
    print(f"\n{model}")

    for metric, value in metrics.items():
        print(f"  {metric}: {value}")

print(f"\nSaved to: {OUTPUT_PATH}")