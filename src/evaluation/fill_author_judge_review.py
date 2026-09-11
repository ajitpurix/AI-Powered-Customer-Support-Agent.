from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]

HUMAN_PATH = BASE_DIR / "data" / "golden" / "judge_human_review.csv"


AUTHOR_SCORES = {
    "2031565": (4, 3, 4),
    "1339673": (5, 4, 5),
    "1575030": (4, 2, 5),
    "2136653": (5, 4, 5),
    "2774298": (5, 4, 5),
    "39686": (4, 3, 5),
    "2510983": (5, 2, 5),
    "1921884": (5, 3, 5),
    "2304557": (5, 4, 5),
    "1397341": (5, 2, 5),
    "2282184": (5, 4, 5),
    "1410770": (5, 3, 5),
    "2296308": (5, 3, 5),
    "1552181": (5, 3, 5),
    "977654": (5, 4, 5),
    "755743": (5, 3, 5),
    "1584602": (5, 4, 5),
    "486291": (4, 2, 4),
    "2700945": (4, 2, 4),
    "2964189": (5, 4, 5),
}


def main():
    review = pd.read_csv(HUMAN_PATH)

    groundedness = []
    helpfulness = []
    hallucination = []
    notes = []

    for tweet_id in review["customer_tweet_id"].astype(str):
        scores = AUTHOR_SCORES[tweet_id]
        groundedness.append(scores[0])
        helpfulness.append(scores[1])
        hallucination.append(scores[2])
        notes.append(
            "Author review on the same 1-5 rubric: groundedness checks support from evidence, "
            "helpfulness checks issue fit, and hallucination checks unsupported claims."
        )

    review["groundedness"] = groundedness
    review["helpfulness"] = helpfulness
    review["hallucination"] = hallucination
    review["reviewer_notes"] = notes

    review.to_csv(HUMAN_PATH, index=False)

    print(f"Filled {len(review)} author-review rows in {HUMAN_PATH}")


if __name__ == "__main__":
    main()
