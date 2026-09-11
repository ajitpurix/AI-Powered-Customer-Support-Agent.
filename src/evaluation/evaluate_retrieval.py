from pathlib import Path
import sys
import json

import pandas as pd


# ============================================================
# PATH SETUP
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(BASE_DIR))

from src.retrieval.retriever import HistoricalRetriever


GOLDEN_PATH = (
    BASE_DIR
    / "data"
    / "golden"
    / "golden_set_final.csv"
)

RESULT_PATH = (
    BASE_DIR
    / "reports"
    / "retrieval_results.json"
)


# ============================================================
# EVALUATION
# ============================================================

def main():

    print("\n======================================")
    print("       RETRIEVAL EVALUATION")
    print("======================================\n")

    golden = pd.read_csv(GOLDEN_PATH)

    golden["customer_text"] = (
        golden["customer_text"]
        .fillna("")
        .astype(str)
    )

    print(
        f"Golden examples: {len(golden)}"
    )

    print("\nLoading historical retriever...")

    retriever = HistoricalRetriever()

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    total = len(golden)

    found_top1 = 0
    found_top3 = 0
    found_top5 = 0

    similarity_scores = []

    # --------------------------------------------------------
    # Evaluate every golden example
    # --------------------------------------------------------

    for index, row in golden.iterrows():

        query = row["customer_text"]
        tweet_id = row.get(
            "customer_tweet_id",
            None
        )

        results = retriever.search(
            query,
            top_k=5,
            min_similarity=0.20,
            exclude_tweet_ids=(
                [tweet_id]
                if tweet_id is not None
                else None
            )
        )

        if results is None or results.empty:
            continue

        similarities = (
            results["similarity"]
            .astype(float)
            .tolist()
        )

        if not similarities:
            continue

        # Top similarity
        similarity_scores.append(
            similarities[0]
        )

        # We measure retrieval coverage:
        # whether the retriever returns at least one
        # historical case.

        found_top1 += 1

        if len(results) >= 3:
            found_top3 += 1

        if len(results) >= 5:
            found_top5 += 1

    # --------------------------------------------------------
    # Calculate metrics
    # --------------------------------------------------------

    coverage = found_top1 / total if total else 0

    top3_coverage = (
        found_top3 / total
        if total
        else 0
    )

    top5_coverage = (
        found_top5 / total
        if total
        else 0
    )

    mean_similarity = (
        sum(similarity_scores)
        / len(similarity_scores)
        if similarity_scores
        else 0
    )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    results = {

        "retriever": "TF-IDF cosine similarity",

        "golden_examples": total,

        "retrieval_coverage": round(
            coverage,
            4
        ),

        "top3_coverage": round(
            top3_coverage,
            4
        ),

        "top5_coverage": round(
            top5_coverage,
            4
        ),

        "mean_top_similarity": round(
            mean_similarity,
            4
        ),

        "leakage_control": (
            "The query tweet_id is excluded from the retrieval index "
            "before ranking, so golden examples cannot retrieve the "
            "exact same row as evidence."
        ),
    }

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    RESULT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        RESULT_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            results,
            file,
            indent=2
        )

    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print("\n======================================")
    print("              RESULTS")
    print("======================================")

    print(
        f"\nRetrieval coverage: "
        f"{coverage:.4f}"
    )

    print(
        f"Top-3 coverage: "
        f"{top3_coverage:.4f}"
    )

    print(
        f"Top-5 coverage: "
        f"{top5_coverage:.4f}"
    )

    print(
        f"Mean top similarity: "
        f"{mean_similarity:.4f}"
    )

    print(
        f"\nResults saved to:"
        f"\n{RESULT_PATH}"
    )


if __name__ == "__main__":
    main()
