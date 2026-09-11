from pathlib import Path
import json

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

TRAIN_PATH = (
    BASE_DIR
    / "data"
    / "golden"
    / "train.csv"
)

GOLDEN_PATH = (
    BASE_DIR
    / "data"
    / "golden"
    / "golden_set_final.csv"
)

PREDICTIONS_PATH = (
    BASE_DIR
    / "data"
    / "golden"
    / "svm_predictions.csv"
)

REPORT_PATH = (
    BASE_DIR
    / "reports"
    / "intent_results.json"
)


# ============================================================
# FIND LABEL COLUMN
# ============================================================

def find_label_column(df):

    possible_columns = [
        "human_intent",
        "gold_intent",
        "suggested_intent",
        "intent",
    ]

    for column in possible_columns:

        if column in df.columns:
            return column

    raise ValueError(
        "No intent label column found."
    )


# ============================================================
# MAIN EVALUATION
# ============================================================

def main():

    print("\n======================================")
    print("       INTENT CLASSIFIER EVALUATION")
    print("======================================\n")


    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    train = pd.read_csv(TRAIN_PATH)
    golden = pd.read_csv(GOLDEN_PATH)


    train_label_column = find_label_column(train)
    golden_label_column = find_label_column(golden)


    train["customer_text"] = (
        train["customer_text"]
        .fillna("")
        .astype(str)
    )

    golden["customer_text"] = (
        golden["customer_text"]
        .fillna("")
        .astype(str)
    )


    y_train = (
        train[train_label_column]
        .fillna("")
        .astype(str)
    )

    y_test = (
        golden[golden_label_column]
        .fillna("")
        .astype(str)
    )


    # --------------------------------------------------------
    # TF-IDF
    # --------------------------------------------------------

    print("Training TF-IDF vectorizer...")

    vectorizer = TfidfVectorizer(

        stop_words="english",

        ngram_range=(1, 2),

        min_df=2,

        max_df=0.95,

        sublinear_tf=True,
    )


    X_train = vectorizer.fit_transform(
        train["customer_text"]
    )

    X_test = vectorizer.transform(
        golden["customer_text"]
    )


    # --------------------------------------------------------
    # Linear SVM
    # --------------------------------------------------------

    print("Training Linear SVM...")

    classifier = LinearSVC(

        C=1.0,

        class_weight="balanced",
    )


    classifier.fit(
        X_train,
        y_train
    )


    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    print("Generating predictions...")

    predictions = classifier.predict(
        X_test
    )


    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        predictions
    )

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


    # --------------------------------------------------------
    # Classification report
    # --------------------------------------------------------

    report = classification_report(

        y_test,

        predictions,

        output_dict=True,

        zero_division=0
    )


    # --------------------------------------------------------
    # Save predictions
    # --------------------------------------------------------

    prediction_df = golden.copy()

    prediction_df["predicted_intent"] = predictions

    prediction_df.to_csv(
        PREDICTIONS_PATH,
        index=False
    )


    # --------------------------------------------------------
    # Save metrics
    # --------------------------------------------------------

    results = {

        "model": "TF-IDF + Linear SVM",

        "train_examples": len(train),

        "golden_examples": len(golden),

        "accuracy": round(
            float(accuracy),
            4
        ),

        "macro_f1": round(
            float(macro_f1),
            4
        ),

        "weighted_f1": round(
            float(weighted_f1),
            4
        ),

        "classification_report": report,
    }


    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )


    with open(
        REPORT_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            results,
            file,
            indent=2
        )


    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print("\n======================================")
    print("              RESULTS")
    print("======================================")

    print(
        f"\nAccuracy:    {accuracy:.4f}"
    )

    print(
        f"Macro F1:    {macro_f1:.4f}"
    )

    print(
        f"Weighted F1: {weighted_f1:.4f}"
    )


    print("\nClassification Report:\n")

    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0
        )
    )


    print(
        f"\nPredictions saved to:"
        f"\n{PREDICTIONS_PATH}"
    )

    print(
        f"\nResults saved to:"
        f"\n{REPORT_PATH}"
    )


if __name__ == "__main__":
    main()