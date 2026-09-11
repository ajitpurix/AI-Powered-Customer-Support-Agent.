from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]

PREDICTIONS_PATH = (
    BASE_DIR
    / "data"
    / "golden"
    / "svm_predictions.csv"
)

OUTPUT_PATH = (
    BASE_DIR
    / "reports"
    / "intent_failures.csv"
)


def main():

    print("\n======================================")
    print("          FAILURE ANALYSIS")
    print("======================================\n")

    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError(
            f"Missing predictions file:\n{PREDICTIONS_PATH}\n\n"
            "Run evaluate_intents.py first."
        )

    df = pd.read_csv(PREDICTIONS_PATH)

    # Detect actual label column
    if "human_intent" in df.columns:
        gold_column = "human_intent"
    elif "gold_intent" in df.columns:
        gold_column = "gold_intent"
    elif "suggested_intent" in df.columns:
        gold_column = "suggested_intent"
    else:
        raise ValueError(
            "Could not find gold intent column."
        )

    # Keep only incorrect predictions
    failures = df[
        df[gold_column].astype(str)
        != df["predicted_intent"].astype(str)
    ].copy()

    print(
        f"Total examples: {len(df)}"
    )

    print(
        f"Incorrect predictions: {len(failures)}"
    )

    if len(df) > 0:
        error_rate = len(failures) / len(df)

        print(
            f"Error rate: {error_rate:.4f}"
        )

    # ------------------------------------------------------
    # Save failures
    # ------------------------------------------------------

    columns = [
        "customer_text",
        gold_column,
        "predicted_intent",
    ]

    available_columns = [
        column
        for column in columns
        if column in failures.columns
    ]

    failures[
        available_columns
    ].to_csv(
        OUTPUT_PATH,
        index=False
    )

    print(
        f"\nFailure examples saved to:\n"
        f"{OUTPUT_PATH}"
    )

    # ------------------------------------------------------
    # Show common confusion pairs
    # ------------------------------------------------------

    if not failures.empty:

        print(
            "\n======================================"
        )

        print(
            "       COMMON CONFUSION PAIRS"
        )

        print(
            "======================================\n"
        )

        confusion = (
            failures
            .groupby(
                [gold_column, "predicted_intent"]
            )
            .size()
            .reset_index(
                name="count"
            )
            .sort_values(
                "count",
                ascending=False
            )
        )

        print(
            confusion.head(10).to_string(
                index=False
            )
        )

        # --------------------------------------------------
        # Show real examples
        # --------------------------------------------------

        print(
            "\n======================================"
        )

        print(
            "          REAL FAILURE EXAMPLES"
        )

        print(
            "======================================\n"
        )

        for i, (_, row) in enumerate(
            failures.head(10).iterrows(),
            start=1
        ):

            print(
                f"--- Failure {i} ---"
            )

            print(
                f"Customer: {row['customer_text']}"
            )

            print(
                f"Expected: {row[gold_column]}"
            )

            print(
                f"Predicted: {row['predicted_intent']}"
            )

            print()


if __name__ == "__main__":
    main()