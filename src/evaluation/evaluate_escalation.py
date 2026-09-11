from pathlib import Path
import sys
import json

import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


# ============================================================
# PATH SETUP
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(BASE_DIR))

from src.retrieval.retriever import HistoricalRetriever
from src.escalation.decision import decide


GOLDEN_PATH = (
    BASE_DIR
    / "data"
    / "golden"
    / "golden_set_final.csv"
)

RESULT_PATH = (
    BASE_DIR
    / "reports"
    / "escalation_results.json"
)


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n======================================")
    print("        ESCALATION EVALUATION")
    print("======================================\n")

    golden = pd.read_csv(GOLDEN_PATH)

    golden["customer_text"] = (
        golden["customer_text"]
        .fillna("")
        .astype(str)
    )

    print(
        f"Evaluation examples: {len(golden)}"
    )

    print("\nLoading historical retriever...")

    retriever = HistoricalRetriever()

    actual_decisions = []
    expected_decisions = []
    examples = []

    # ========================================================
    # Evaluate
    # ========================================================

    for index, row in golden.iterrows():

        message = row["customer_text"]
        tweet_id = row.get(
            "customer_tweet_id",
            None
        )

        # Retrieve evidence
        evidence = retriever.search(
            message,
            top_k=5,
            min_similarity=0.20,
            exclude_tweet_ids=(
                [tweet_id]
                if tweet_id is not None
                else None
            )
        )

        # Agent decision
        decision, reason = decide(
            message,
            evidence
        )

        # ----------------------------------------------------
        # IMPORTANT:
        # This is a proxy evaluation target.
        # It is NOT human-labelled escalation ground truth.
        # ----------------------------------------------------

        intent = str(
            row.get("human_intent", "")
        )

        if not intent:
            intent = str(
                row.get("gold_intent", "")
            )

        # High-risk intents are expected to escalate
        if intent in [
            "billing_refund",
            "apple_id_account",
        ]:
            expected = "escalate"
        else:
            expected = "auto_handle"

        actual_decisions.append(
            decision
        )

        expected_decisions.append(
            expected
        )

        examples.append({
            "customer_text": message,
            "intent": intent,
            "expected_decision": expected,
            "predicted_decision": decision,
            "reason": reason,
        })

    # ========================================================
    # Metrics
    # ========================================================

    accuracy = accuracy_score(
        expected_decisions,
        actual_decisions
    )

    precision = precision_score(
        expected_decisions,
        actual_decisions,
        pos_label="escalate",
        zero_division=0
    )

    recall = recall_score(
        expected_decisions,
        actual_decisions,
        pos_label="escalate",
        zero_division=0
    )

    f1 = f1_score(
        expected_decisions,
        actual_decisions,
        pos_label="escalate",
        zero_division=0
    )

    cm = confusion_matrix(
        expected_decisions,
        actual_decisions,
        labels=[
            "auto_handle",
            "escalate"
        ]
    )

    # ========================================================
    # Results
    # ========================================================

    results = {

        "model": "Rule-based escalation",

        "evaluation_examples": len(golden),

        "accuracy": round(
            float(accuracy),
            4
        ),

        "escalation_precision": round(
            float(precision),
            4
        ),

        "escalation_recall": round(
            float(recall),
            4
        ),

        "escalation_f1": round(
            float(f1),
            4
        ),

        "confusion_matrix": {
            "labels": [
                "auto_handle",
                "escalate"
            ],
            "matrix": cm.tolist()
        },

        "evaluation_warning": (
            "Expected escalation labels are proxy labels "
            "derived from intent categories. They are NOT "
            "independently human-labelled escalation ground truth."
        ),

        "examples": examples
    }

    # ========================================================
    # Save
    # ========================================================

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
            indent=2,
            ensure_ascii=False
        )

    # ========================================================
    # Print
    # ========================================================

    print("\n======================================")
    print("              RESULTS")
    print("======================================")

    print(
        f"\nAccuracy: "
        f"{accuracy:.4f}"
    )

    print(
        f"Escalation precision: "
        f"{precision:.4f}"
    )

    print(
        f"Escalation recall: "
        f"{recall:.4f}"
    )

    print(
        f"Escalation F1: "
        f"{f1:.4f}"
    )

    print("\nWARNING:")
    print(
        "These metrics use proxy escalation labels, "
        "not human-labelled ground truth."
    )

    print(
        f"\nResults saved to:\n"
        f"{RESULT_PATH}"
    )


if __name__ == "__main__":
    main()
