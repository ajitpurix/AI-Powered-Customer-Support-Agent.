from pathlib import Path
import sys

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(BASE_DIR))

from src.retrieval.retriever import HistoricalRetriever
from src.generation.reply_generator import ReplyGenerator
from src.escalation.decision import decide


TRAIN_PATH = BASE_DIR / "data" / "golden" / "train.csv"


class SupportAgent:

    def __init__(self):
        print("\nLoading training data...")

        self.train_data = pd.read_csv(TRAIN_PATH)

        # Automatically detect the label column
        possible_labels = [
            "human_intent",
            "gold_intent",
            "suggested_intent",
            "intent",
        ]

        label_column = None

        for column in possible_labels:
            if column in self.train_data.columns:
                label_column = column
                break

        if label_column is None:
            raise ValueError(
                "Could not find intent label column in train.csv"
            )

        self.label_column = label_column

        self.train_data["customer_text"] = (
            self.train_data["customer_text"]
            .fillna("")
            .astype(str)
        )

        self.train_data[label_column] = (
            self.train_data[label_column]
            .fillna("")
            .astype(str)
        )

        # -----------------------------
        # Intent classifier
        # -----------------------------

        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            sublinear_tf=True,
        )

        X_train = self.vectorizer.fit_transform(
            self.train_data["customer_text"]
        )

        self.classifier = LinearSVC(
            C=1.0,
            class_weight="balanced",
        )

        self.classifier.fit(
            X_train,
            self.train_data[label_column],
        )

        print(
            f"Intent classifier trained on "
            f"{len(self.train_data)} examples."
        )

        # -----------------------------
        # Historical retrieval
        # -----------------------------

        self.retriever = HistoricalRetriever()

        # -----------------------------
        # Reply generation
        # -----------------------------

        self.reply_generator = ReplyGenerator()

        print("Support agent initialized successfully.")


    def classify_intent(self, customer_message):

        vector = self.vectorizer.transform(
            [customer_message]
        )

        intent = self.classifier.predict(vector)[0]

        return intent


    def process(self, customer_message):

        # 1. Classify intent
        intent = self.classify_intent(
            customer_message
        )

        # 2. Retrieve historical evidence
        evidence = self.retriever.search(
            customer_message,
            top_k=5,
            min_similarity=0.20,
        )

        # 3. Decide auto-handle vs escalation
        decision, reason = decide(
            customer_message,
            evidence,
        )

        # 4. Generate grounded reply
        generation_result = self.reply_generator.generate(
            customer_message,
            evidence,
        )

        return {
            "customer_message": customer_message,
            "intent": intent,
            "reply": generation_result["reply"],
            "decision": decision,
            "reason": reason,
            "evidence": evidence,
            "generation_mode": generation_result["mode"],
        }


def main():

    print("\n======================================")
    print("      APPLE SUPPORT AI AGENT")
    print("======================================")

    agent = SupportAgent()

    customer_message = input(
        "\nEnter customer message: "
    ).strip()

    if not customer_message:
        print("Customer message cannot be empty.")
        return

    result = agent.process(
        customer_message
    )

    print("\n======================================")
    print("             RESULT")
    print("======================================")

    print(
        f"\nCustomer:\n"
        f"{result['customer_message']}"
    )

    print(
        f"\nIntent:\n"
        f"{result['intent']}"
    )

    print(
        f"\nDecision:\n"
        f"{result['decision']}"
    )

    print(
        f"\nReason:\n"
        f"{result['reason']}"
    )

    print(
        f"\nGenerated Reply:\n"
        f"{result['reply']}"
    )

    print("\nHistorical Evidence:")

    evidence = result["evidence"]

    if evidence is None or evidence.empty:
        print("No historical evidence found.")

    else:
        for i, row in evidence.iterrows():

            print(
                f"\n--- Evidence {i + 1} ---"
            )

            print(
                f"Similarity: "
                f"{row['similarity']:.4f}"
            )

            print(
                f"Customer: "
                f"{row['customer_text']}"
            )

            print(
                f"Historical Reply: "
                f"{row['reply_text']}"
            )


if __name__ == "__main__":
    main()