from pathlib import Path
import json
import sys

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(BASE_DIR))

from src.evaluation.judge import ReplyJudge


REPLY_RESULTS_PATH = (
    BASE_DIR
    / "reports"
    / "reply_results.json"
)

OUTPUT_PATH = (
    BASE_DIR
    / "data"
    / "golden"
    / "llm_judge_results.csv"
)


def local_rubric_judge(item):
    evidence = item.get("evidence", [])
    generated_reply = item.get("generated_reply", "")
    top_similarity = float(item.get("top_similarity", 0) or 0)

    if not evidence:
        return {
            "groundedness": 3,
            "helpfulness": 2,
            "hallucination": 5,
            "reason": "No evidence was retrieved; the reply appropriately escalates instead of inventing an answer.",
            "judge_source": "local_rubric_fallback",
        }

    top_reply = str(evidence[0].get("reply_text", "")).strip()
    reply_contains_evidence = top_reply and top_reply in generated_reply

    if reply_contains_evidence and top_similarity >= 0.35:
        groundedness = 5
    elif reply_contains_evidence:
        groundedness = 4
    else:
        groundedness = 3

    if top_similarity >= 0.50:
        helpfulness = 4
    elif top_similarity >= 0.30:
        helpfulness = 3
    else:
        helpfulness = 2

    risky_terms = [
        "refund",
        "replacement",
        "credit",
        "guarantee",
        "guaranteed",
    ]
    generated_lower = generated_reply.lower()
    evidence_text = " ".join(
        str(row.get("reply_text", ""))
        for row in evidence
    ).lower()

    unsupported_risky_claim = any(
        term in generated_lower and term not in evidence_text
        for term in risky_terms
    )

    if unsupported_risky_claim:
        hallucination = 3
    elif top_similarity < 0.30:
        hallucination = 4
    else:
        hallucination = 5

    return {
        "groundedness": groundedness,
        "helpfulness": helpfulness,
        "hallucination": hallucination,
        "reason": (
            "Deterministic rubric fallback: checked whether the reply reused retrieved evidence, "
            "whether top similarity was strong enough to be useful, and whether unsupported risky "
            "claims appeared."
        ),
        "judge_source": "local_rubric_fallback",
    }


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

    judge = None
    judge_source = "openai"

    try:
        judge = ReplyJudge()
    except ValueError:
        judge_source = "local_rubric_fallback"
        print(
            "OPENAI_API_KEY not found. Using local rubric fallback "
            "so the agreement harness remains reproducible."
        )
    rows = []

    for index, item in enumerate(
        reply_results.get("examples", []),
        start=1
    ):
        evidence = pd.DataFrame(
            item.get("evidence", [])
        )

        print(
            f"Judging reply {index}..."
        )

        if judge is not None:
            result = judge.judge(
                customer_message=item.get(
                    "customer_text",
                    ""
                ),
                historical_evidence=evidence,
                generated_reply=item.get(
                    "generated_reply",
                    ""
                ),
            )
            result["judge_source"] = judge_source
        else:
            result = local_rubric_judge(item)

        rows.append({
            "customer_tweet_id": item.get(
                "customer_tweet_id",
                ""
            ),
            "groundedness": result.get(
                "groundedness"
            ),
            "helpfulness": result.get(
                "helpfulness"
            ),
            "hallucination": result.get(
                "hallucination"
            ),
            "reason": result.get(
                "reason",
                ""
            ),
            "judge_source": result.get(
                "judge_source",
                judge_source
            ),
        })

    output = pd.DataFrame(rows)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    output.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print(
        f"Saved {len(output)} LLM judge rows to {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()
