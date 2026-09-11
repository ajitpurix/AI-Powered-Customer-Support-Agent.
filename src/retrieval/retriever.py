from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# --------------------------------------------------
# Project paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[2]

CONVERSATIONS_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "apple_conversations.csv"
)


# --------------------------------------------------
# Historical Retriever
# --------------------------------------------------

class HistoricalRetriever:

    def __init__(
        self,
        conversations_path=CONVERSATIONS_PATH
    ):

        # Load historical conversations
        self.data = pd.read_csv(
            conversations_path
        )

        # Clean customer messages
        self.data["customer_text"] = (
            self.data["customer_text"]
            .fillna("")
            .astype(str)
        )

        # Clean support replies
        self.data["reply_text"] = (
            self.data["reply_text"]
            .fillna("")
            .astype(str)
        )

        # TF-IDF configuration
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            min_df=2,
            max_features=50000,
            sublinear_tf=True
        )

        # Build TF-IDF vectors
        self.customer_vectors = (
            self.vectorizer.fit_transform(
                self.data["customer_text"]
            )
        )

        print(
            f"Loaded {len(self.data)} historical "
            "support conversations."
        )

    # --------------------------------------------------
    # Search
    # --------------------------------------------------

    def search(
        self,
        query,
        top_k=5,
        min_similarity=0.20,
        exclude_tweet_ids=None
    ):
        """
        Retrieve the most similar historical
        customer-support conversations.

        Parameters
        ----------
        query : str
            New customer message.

        top_k : int
            Maximum number of results.

        min_similarity : float
            Minimum cosine similarity required.

        Returns
        -------
        pandas.DataFrame
        """

        # Convert query to TF-IDF
        query_vector = (
            self.vectorizer.transform(
                [query]
            )
        )

        # Calculate cosine similarity
        scores = cosine_similarity(
            query_vector,
            self.customer_vectors
        ).flatten()

        # Optionally remove known evaluation rows before ranking.
        # This prevents golden examples from retrieving themselves.
        if exclude_tweet_ids:
            excluded = {
                str(tweet_id)
                for tweet_id in exclude_tweet_ids
            }

            mask = (
                self.data["customer_tweet_id"]
                .astype(str)
                .isin(excluded)
            )

            scores[mask.to_numpy()] = -1.0

        # Get top results
        top_indices = (
            scores.argsort()[::-1][:top_k]
        )

        results = self.data.iloc[
            top_indices
        ].copy()

        # Add similarity
        results["similarity"] = (
            scores[top_indices]
        )

        # Remove weak matches
        results = results[
            results["similarity"] >= min_similarity
        ]

        # Return required columns
        return results[
            [
                "customer_tweet_id",
                "customer_text",
                "reply_text",
                "similarity"
            ]
        ].reset_index(drop=True)


# --------------------------------------------------
# Manual testing
# --------------------------------------------------

if __name__ == "__main__":

    retriever = HistoricalRetriever()

    query = input(
        "\nEnter a customer message: "
    ).strip()

    if not query:

        print(
            "Customer message cannot be empty."
        )

    else:

        results = retriever.search(
            query,
            top_k=5,
            min_similarity=0.20
        )

        print(
            "\n===== SIMILAR HISTORICAL CASES =====\n"
        )

        if results.empty:

            print(
                "No sufficiently similar "
                "historical cases found."
            )

        else:

            for i, row in results.iterrows():

                print(
                    f"--- Result {i + 1} ---"
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
                    f"Reply: "
                    f"{row['reply_text']}"
                )

                print()
