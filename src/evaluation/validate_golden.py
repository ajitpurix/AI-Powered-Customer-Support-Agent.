from pathlib import Path
import pandas as pd


INPUT_PATH = "data/golden/golden_review.csv"
OUTPUT_PATH = "data/golden/golden_set_final.csv"


VALID_INTENTS = {
    "ios_software_issue",
    "battery_power_issue",
    "device_hardware_issue",
    "connectivity_issue",
    "apple_id_account",
    "icloud_issue",
    "app_service_issue",
    "app_store_purchase",
    "billing_refund",
    "general_inquiry",
}


def main():

    df = pd.read_csv(INPUT_PATH)

    # Check required column
    if "gold_intent" not in df.columns:
        raise ValueError(
            "gold_intent column is missing."
        )

    # Check for empty labels
    empty = (
        df["gold_intent"]
        .isna()
        |
        (df["gold_intent"].astype(str).str.strip() == "")
    )

    if empty.any():
        print(
            f"ERROR: {empty.sum()} examples "
            "still need human labels."
        )
        return

    # Check invalid labels
    invalid = ~df["gold_intent"].isin(
        VALID_INTENTS
    )

    if invalid.any():

        print("ERROR: Invalid intent labels found:")

        print(
            df.loc[
                invalid,
                "gold_intent"
            ].unique()
        )

        return

    # Create final Golden Set
    final = df[
        [
            "customer_tweet_id",
            "customer_text",
            "reply_text",
            "gold_intent",
        ]
    ].copy()

    final = final.rename(
        columns={
            "gold_intent": "human_intent"
        }
    )

    final.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("\n===== GOLDEN SET VALID =====")

    print(
        f"Examples: {len(final)}"
    )

    print("\nDistribution:")

    print(
        final["human_intent"]
        .value_counts()
    )

    print(
        f"\nSaved to: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()