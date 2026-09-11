import pandas as pd
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

TRAIN_PATH = Path("data/golden/train.csv")
GOLDEN_PATH = Path("data/golden/golden_set.csv")


print("Loading datasets...")

train_df = pd.read_csv(TRAIN_PATH)
golden_df = pd.read_csv(GOLDEN_PATH)

X_train = train_df["customer_text"].astype(str)
y_train = train_df["human_intent"].astype(str)

X_test = golden_df["customer_text"].astype(str)
y_test = golden_df["human_intent"].astype(str)

print(f"Training examples: {len(train_df)}")
print(f"Golden examples:   {len(golden_df)}")


# ============================================================
# TF-IDF + LOGISTIC REGRESSION
# ============================================================

model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=1,
            max_features=20000,
        ),
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
        ),
    ),
])


print("\nTraining TF-IDF + Logistic Regression...")

model.fit(X_train, y_train)

predictions = model.predict(X_test)


# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(y_test, predictions)
macro_f1 = f1_score(
    y_test,
    predictions,
    average="macro",
    zero_division=0,
)
weighted_f1 = f1_score(
    y_test,
    predictions,
    average="weighted",
    zero_division=0,
)

print("\n" + "=" * 70)
print("TF-IDF + LOGISTIC REGRESSION RESULTS")
print("=" * 70)

print(f"Accuracy:    {accuracy:.4f}")
print(f"Macro F1:    {macro_f1:.4f}")
print(f"Weighted F1: {weighted_f1:.4f}")


# ============================================================
# PER-INTENT RESULTS
# ============================================================

print("\n===== CLASSIFICATION REPORT =====")

print(
    classification_report(
        y_test,
        predictions,
        zero_division=0,
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

labels = sorted(y_test.unique())

cm = confusion_matrix(
    y_test,
    predictions,
    labels=labels,
)

print("\n===== CONFUSION MATRIX =====")

print("Labels:")
print(labels)

print("\nMatrix:")
print(cm)


# ============================================================
# SHOW MISTAKES
# ============================================================

results = golden_df[
    ["customer_tweet_id", "customer_text", "human_intent"]
].copy()

results["predicted_intent"] = predictions

errors = results[
    results["human_intent"] != results["predicted_intent"]
]

print("\n===== EXAMPLE ERRORS =====")

for _, row in errors.head(15).iterrows():
    print("\nCustomer:")
    print(row["customer_text"])

    print("Actual:   ", row["human_intent"])
    print("Predicted:", row["predicted_intent"])


# ============================================================
# SAVE PREDICTIONS
# ============================================================

output_path = Path("data/golden/tfidf_predictions.csv")

results.to_csv(
    output_path,
    index=False,
)

print(f"\nPredictions saved to: {output_path}")