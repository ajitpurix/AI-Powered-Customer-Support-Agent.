from pathlib import Path
import pandas as pd


INPUT_PATH = "data/golden/golden_set.csv"
OUTPUT_PATH = "data/golden/golden_review.csv"


def main():

    df = pd.read_csv(INPUT_PATH)

    review = df[
        [
            "customer_tweet_id",
            "customer_text",
            "reply_text",
            "human_intent"
        ]
    ].copy()

    # Rename provisional label so we don't
    # accidentally treat it as ground truth.
    review = review.rename(
        columns={
            "human_intent": "suggested_intent"
        }
    )

    # Human reviewer must fill this column.
    review["gold_intent"] = ""

    review["review_status"] = "needs_review"

    review.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print(
        f"Created {len(review)} examples for human review."
    )

    print(
        f"Saved to: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()