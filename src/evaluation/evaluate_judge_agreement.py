from pathlib import Path
import json

import pandas as pd

from sklearn.metrics import cohen_kappa_score


BASE_DIR = Path(__file__).resolve().parents[2]

HUMAN_PATH = (
    BASE_DIR
    / "data"
    / "golden"
    / "judge_human_review.csv"
)

LLM_PATH = (
    BASE_DIR
    / "data"
    / "golden"
    / "llm_judge_results.csv"
)

REPORT_PATH = (
    BASE_DIR
    / "reports"
    / "judge_agreement_results.json"
)

DIMENSIONS = [
    "groundedness",
    "helpfulness",
    "hallucination",
]


def require_columns(df, columns, path):
    missing = [
        column
        for column in columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{path} is missing columns: {missing}"
        )


def main():
    if not HUMAN_PATH.exists():
        raise FileNotFoundError(
            f"Missing human review file: {HUMAN_PATH}\n"
            "Create it from reply_results.json and score each reply "
            "from 1 to 5 for groundedness, helpfulness, and hallucination."
        )

    if not LLM_PATH.exists():
        raise FileNotFoundError(
            f"Missing LLM judge file: {LLM_PATH}\n"
            "Run the LLM judge on the same examples and save one row "
            "per customer_tweet_id with the same score columns."
        )

    human = pd.read_csv(HUMAN_PATH)
    llm = pd.read_csv(LLM_PATH)

    required = [
        "customer_tweet_id",
        *DIMENSIONS,
    ]

    require_columns(
        human,
        required,
        HUMAN_PATH
    )

    require_columns(
        llm,
        required,
        LLM_PATH
    )

    merged = human.merge(
        llm,
        on="customer_tweet_id",
        suffixes=("_human", "_llm"),
    )

    if merged.empty:
        raise ValueError(
            "No overlapping customer_tweet_id values found."
        )

    dimension_results = {}

    for dimension in DIMENSIONS:
        human_scores = (
            merged[f"{dimension}_human"]
            .astype(int)
        )

        llm_scores = (
            merged[f"{dimension}_llm"]
            .astype(int)
        )

        exact_agreement = (
            human_scores == llm_scores
        ).mean()

        within_one = (
            (human_scores - llm_scores)
            .abs()
            <= 1
        ).mean()

        kappa = cohen_kappa_score(
            human_scores,
            llm_scores,
            weights="quadratic",
        )

        dimension_results[dimension] = {
            "exact_agreement": round(
                float(exact_agreement),
                4
            ),
            "within_one_agreement": round(
                float(within_one),
                4
            ),
            "quadratic_weighted_kappa": round(
                float(kappa),
                4
            ),
        }

    results = {
        "examples": len(merged),
        "dimensions": dimension_results,
        "note": (
            "Agreement is calculated on the intersection of human "
            "review rows and LLM judge rows by customer_tweet_id."
        ),
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

    print(
        json.dumps(
            results,
            indent=2
        )
    )


if __name__ == "__main__":
    main()
