from pathlib import Path
import json

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]

REPLY_RESULTS_PATH = (
    BASE_DIR
    / "reports"
    / "reply_results.json"
)

OUTPUT_PATH = (
    BASE_DIR
    / "data"
    / "golden"
    / "judge_human_review.csv"
)


def main():
    if not REPLY_RESULTS_PATH.exists():
        raise FileNotFoundError(
            f"Missing reply results file: {REPLY_RESULTS_PATH}\n"
            "Run src/evaluation/evaluate_replies.py first."
        )

    with open(
        REPLY_RESULTS_PATH,
        "r",
        encoding="utf-8"
    ) as file:
        reply_results = json.load(file)

    rows = []

    for item in reply_results.get("examples", []):
        evidence = item.get("evidence", [])

        rows.append({
            "customer_tweet_id": item.get(
                "customer_tweet_id",
                ""
            ),
            "customer_text": item.get(
                "customer_text",
                ""
            ),
            "generated_reply": item.get(
                "generated_reply",
                ""
            ),
            "top_evidence_reply": (
                evidence[0].get("reply_text", "")
                if evidence
                else ""
            ),
            "groundedness": "",
            "helpfulness": "",
            "hallucination": "",
            "reviewer_notes": "",
        })

    review = pd.DataFrame(rows)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    review.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print(
        f"Created {len(review)} human judge-review rows."
    )

    print(
        f"Saved to: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()
