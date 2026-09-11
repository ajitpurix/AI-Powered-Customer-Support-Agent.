from pathlib import Path
import sys
import json

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(BASE_DIR))

from src.retrieval.retriever import HistoricalRetriever
from src.generation.reply_generator import ReplyGenerator


EVAL_PATH = (
    BASE_DIR
    / "data"
    / "golden"
    / "reply_eval.csv"
)

RESULT_PATH = (
    BASE_DIR
    / "reports"
    / "reply_results.json"
)


def main():

    print("\n======================================")
    print("         REPLY QUALITY EVALUATION")
    print("======================================\n")

    # --------------------------------------------------
    # Load evaluation examples
    # --------------------------------------------------

    if not EVAL_PATH.exists():

        raise FileNotFoundError(
            f"Evaluation file not found:\n{EVAL_PATH}\n\n"
            "Create it first using:\n"
            "python -c \"import pandas as pd; "
            "df=pd.read_csv('data/golden/golden_set_final.csv'); "
            "df.sample(n=min(20,len(df)), random_state=42)"
            ".to_csv('data/golden/reply_eval.csv', index=False)\""
        )

    eval_data = pd.read_csv(EVAL_PATH)

    eval_data["customer_text"] = (
        eval_data["customer_text"]
        .fillna("")
        .astype(str)
    )

    print(
        f"Evaluation examples: {len(eval_data)}"
    )

    # --------------------------------------------------
    # Create ONE retriever
    # --------------------------------------------------

    print("\nLoading historical retriever...")

    retriever = HistoricalRetriever()

    # Pass the same retriever to generator
    generator = ReplyGenerator(
        retriever
    )

    results = []

    # --------------------------------------------------
    # Evaluate each example
    # --------------------------------------------------

    for index, row in eval_data.iterrows():

        customer_message = row["customer_text"]
        tweet_id = row.get(
            "customer_tweet_id",
            None
        )

        print(
            f"\nEvaluating "
            f"{index + 1}/{len(eval_data)}..."
        )

        evidence = retriever.search(
            customer_message,
            top_k=5,
            min_similarity=0.20,
            exclude_tweet_ids=(
                [tweet_id]
                if tweet_id is not None
                else None
            )
        )

        generated = generator.generate(
            customer_message,
            evidence
        )

        # Save a compact representation of evidence
        evidence_records = []

        if evidence is not None and not evidence.empty:

            for _, evidence_row in evidence.iterrows():

                evidence_records.append({
                    "customer_text": str(
                        evidence_row["customer_text"]
                    ),
                    "reply_text": str(
                        evidence_row["reply_text"]
                    ),
                    "similarity": round(
                        float(
                            evidence_row["similarity"]
                        ),
                        4
                    )
                })

        results.append({

            "customer_tweet_id": str(
                row.get(
                    "customer_tweet_id",
                    ""
                )
            ),

            "customer_text": customer_message,

            "generated_reply": generated["reply"],

            "generation_mode": generated["mode"],

            "evidence": evidence_records,

            "top_similarity": (
                evidence_records[0]["similarity"]
                if evidence_records
                else 0.0
            ),

            "evidence_found": bool(
                evidence_records
            )
        })

    # --------------------------------------------------
    # Summary metrics
    # --------------------------------------------------

    total = len(results)

    evidence_found = sum(
        1
        for item in results
        if item["evidence_found"]
    )

    fallback_count = sum(
        1
        for item in results
        if item["generation_mode"]
        == "local_fallback"
    )

    no_evidence_count = sum(
        1
        for item in results
        if item["generation_mode"]
        == "no_evidence"
    )

    openai_count = sum(
        1
        for item in results
        if item["generation_mode"]
        == "openai"
    )

    mean_similarity = (
        sum(
            item["top_similarity"]
            for item in results
            if item["top_similarity"] > 0
        )
        / evidence_found
        if evidence_found
        else 0
    )

    summary = {

        "evaluation_examples": total,

        "evidence_found": evidence_found,

        "evidence_coverage": round(
            evidence_found / total
            if total
            else 0,
            4
        ),

        "mean_top_similarity": round(
            mean_similarity,
            4
        ),

        "generation_modes": {

            "openai": openai_count,

            "local_fallback": fallback_count,

            "no_evidence": no_evidence_count
        },

        "note": (
            "This evaluation records generated replies "
            "and retrieval evidence. Human or LLM quality "
            "judging should be performed separately."
        ),

        "leakage_control": (
            "The current evaluation excludes each query tweet_id "
            "from retrieval before ranking."
        ),

        "examples": results
    }

    # --------------------------------------------------
    # Save results
    # --------------------------------------------------

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
            summary,
            file,
            indent=2,
            ensure_ascii=False
        )

    # --------------------------------------------------
    # Print summary
    # --------------------------------------------------

    print("\n======================================")
    print("              RESULTS")
    print("======================================")

    print(
        f"\nExamples: "
        f"{total}"
    )

    print(
        f"Evidence coverage: "
        f"{summary['evidence_coverage']:.4f}"
    )

    print(
        f"Mean top similarity: "
        f"{summary['mean_top_similarity']:.4f}"
    )

    print(
        f"OpenAI generations: "
        f"{openai_count}"
    )

    print(
        f"Local fallback generations: "
        f"{fallback_count}"
    )

    print(
        f"No-evidence generations: "
        f"{no_evidence_count}"
    )

    print(
        f"\nResults saved to:\n"
        f"{RESULT_PATH}"
    )


if __name__ == "__main__":
    main()
