import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix


TRAIN_PATH = "data/golden/train.csv"
GOLDEN_PATH = "data/golden/golden_set.csv"
OUTPUT_PATH = "data/golden/svm_predictions.csv"


def main():
    train = pd.read_csv(TRAIN_PATH)
    golden = pd.read_csv(GOLDEN_PATH)

    X_train = train["customer_text"].fillna("")
    y_train = train["human_intent"]

    X_test = golden["customer_text"].fillna("")
    y_test = golden["human_intent"]

    # TF-IDF representation
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Linear SVM
    model = LinearSVC(
        C=1.0,
        class_weight="balanced"
    )

    model.fit(X_train_vec, y_train)

    predictions = model.predict(X_test_vec)

    # Metrics
    accuracy = accuracy_score(y_test, predictions)
    macro_f1 = f1_score(y_test, predictions, average="macro")
    weighted_f1 = f1_score(y_test, predictions, average="weighted")

    print("\n===== TF-IDF + LINEAR SVM =====")
    print(f"Accuracy:    {accuracy:.4f}")
    print(f"Macro F1:    {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")

    print("\n===== CLASSIFICATION REPORT =====")
    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0
        )
    )

    print("\n===== CONFUSION MATRIX =====")
    labels = sorted(y_test.unique())

    cm = confusion_matrix(
        y_test,
        predictions,
        labels=labels
    )

    print(pd.DataFrame(
        cm,
        index=labels,
        columns=labels
    ))

    # Save predictions
    results = golden.copy()
    results["predicted_intent"] = predictions
    results["correct"] = (
        results["human_intent"] == results["predicted_intent"]
    )

    results.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print(f"\nPredictions saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()