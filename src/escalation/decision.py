from typing import Tuple


HIGH_RISK_KEYWORDS = [
    "refund",
    "charged",
    "charge",
    "payment",
    "fraud",
    "stolen",
    "hacked",
    "account locked",
    "can't access account",
    "cannot access account",
    "legal",
    "lawsuit",
    "complaint",
    "security",
]


LOW_CONFIDENCE_THRESHOLD = 0.20


def decide(
    customer_message: str,
    retrieval_results
) -> Tuple[str, str]:

    message = customer_message.lower().strip()

    # High-risk issues
    matched_keywords = [
        keyword
        for keyword in HIGH_RISK_KEYWORDS
        if keyword in message
    ]

    if matched_keywords:

        return (
            "escalate",
            (
                "High-risk issue detected. "
                f"Matched keyword(s): "
                f"{', '.join(matched_keywords)}."
            )
        )

    # No historical evidence
    if (
        retrieval_results is None
        or retrieval_results.empty
    ):

        return (
            "escalate",
            "No sufficiently similar historical support resolution was found."
        )

    # Weak historical evidence
    top_similarity = float(
        retrieval_results.iloc[0]["similarity"]
    )

    if top_similarity < LOW_CONFIDENCE_THRESHOLD:

        return (
            "escalate",
            (
                "Historical evidence is too weak "
                "for automatic handling. "
                f"Top similarity: {top_similarity:.3f}."
            )
        )

    # Safe to auto-handle
    return (
        "auto_handle",
        (
            "Relevant historical support evidence "
            "was found and no high-risk escalation "
            "condition was detected. "
            f"Top similarity: {top_similarity:.3f}."
        )
    )


if __name__ == "__main__":

    print("Escalation module test")

    decision, reason = decide(
        "I was charged for something I didn't buy",
        None
    )

    print("Decision:", decision)
    print("Reason:", reason)