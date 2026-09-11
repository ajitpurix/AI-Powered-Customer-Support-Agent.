from pathlib import Path
import sys
import os

from dotenv import load_dotenv

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


# ============================================================
# PATH SETUP
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(BASE_DIR))


from src.retrieval.retriever import HistoricalRetriever


# Load environment variables from .env
load_dotenv()


# ============================================================
# REPLY GENERATOR
# ============================================================

class ReplyGenerator:

    def __init__(self, retriever=None):

        # Use the retriever created by SupportAgent.
        # This prevents loading the historical dataset twice.
        self.retriever = retriever

        # OpenAI API is optional
        self.api_key = os.getenv("OPENAI_API_KEY")

        self.client = None

        if self.api_key and OpenAI is not None:

            self.client = OpenAI(
                api_key=self.api_key
            )

            print("OpenAI generation enabled.")

        else:

            print(
                "OpenAI API key not found. "
                "Using local evidence-based fallback."
            )


    # ========================================================
    # BUILD OPENAI PROMPT
    # ========================================================

    def build_prompt(
        self,
        customer_message,
        retrieval_results
    ):

        evidence = ""

        for i, row in retrieval_results.iterrows():

            evidence += f"""
--- Historical Case {i + 1} ---

Customer:
{row["customer_text"]}

Historical Support Reply:
{row["reply_text"]}

Similarity:
{row["similarity"]:.4f}
"""

        prompt = f"""
You are a customer-support reply assistant.

Customer message:

{customer_message}


Historical support cases and their resolutions:

{evidence}


Your task is to draft the best possible
customer-facing support reply.


Rules:

1. Ground the response in the historical evidence.

2. Do not invent information that is not supported
   by the historical evidence.

3. Do not promise refunds, replacements, credits,
   or other actions unless the evidence supports it.

4. If the historical evidence is insufficient,
   say that further investigation is required.

5. Be concise, clear, and professional.

6. Do not mention the historical cases.

7. Do not mention that you are an AI.

8. Return ONLY the customer-facing reply.
"""

        return prompt


    # ========================================================
    # LOCAL FALLBACK
    # ========================================================

    def local_fallback(
        self,
        customer_message,
        retrieval_results
    ):

        # No evidence
        if (
            retrieval_results is None
            or retrieval_results.empty
        ):

            return (
                "I'm sorry, but I couldn't find enough "
                "relevant information to provide a reliable "
                "answer. This issue should be reviewed by "
                "a support agent."
            )


        # Get the most similar historical case
        best_case = retrieval_results.iloc[0]


        historical_reply = str(
            best_case["reply_text"]
        ).strip()


        # Historical reply is empty
        if not historical_reply:

            return (
                "I'm sorry, but I couldn't find enough "
                "information to provide a reliable answer. "
                "This issue should be reviewed by a support agent."
            )


        # Evidence-based local response
        return (
            "Thanks for reaching out. Based on similar "
            "support cases, the recommended guidance is:\n\n"
            f"{historical_reply}\n\n"
            "If this does not resolve the issue, "
            "the case should be reviewed by a support agent."
        )


    # ========================================================
    # GENERATE REPLY
    # ========================================================

    def generate(
        self,
        customer_message,
        retrieval_results
    ):

        # ----------------------------------------------------
        # CASE 1: No historical evidence
        # ----------------------------------------------------

        if (
            retrieval_results is None
            or retrieval_results.empty
        ):

            return {
                "reply": (
                    "I'm sorry, but I couldn't find enough "
                    "relevant information to provide a reliable "
                    "answer. This issue should be reviewed by "
                    "a support agent."
                ),

                "evidence": retrieval_results,

                "mode": "no_evidence"
            }


        # ----------------------------------------------------
        # CASE 2: OpenAI available
        # ----------------------------------------------------

        if self.client is not None:

            prompt = self.build_prompt(
                customer_message,
                retrieval_results
            )


            response = self.client.responses.create(

                model="gpt-5.6-luna",

                input=prompt
            )


            reply = response.output_text.strip()


            return {
                "reply": reply,

                "evidence": retrieval_results,

                "mode": "openai"
            }


        # ----------------------------------------------------
        # CASE 3: No API key
        # ----------------------------------------------------

        reply = self.local_fallback(
            customer_message,
            retrieval_results
        )


        return {
            "reply": reply,

            "evidence": retrieval_results,

            "mode": "local_fallback"
        }


# ============================================================
# STANDALONE TEST
# ============================================================

if __name__ == "__main__":

    print(
        "\n===== APPLE SUPPORT REPLY GENERATOR ====="
    )


    customer_message = input(
        "\nEnter a customer message: "
    ).strip()


    if not customer_message:

        print(
            "Customer message cannot be empty."
        )

        sys.exit(0)


    # Create retriever only for standalone testing
    retriever = HistoricalRetriever()


    generator = ReplyGenerator(
        retriever
    )


    # Retrieve historical evidence
    evidence = retriever.search(

        customer_message,

        top_k=5,

        min_similarity=0.20
    )


    # Generate response
    result = generator.generate(

        customer_message,

        evidence
    )


    # ========================================================
    # OUTPUT
    # ========================================================

    print(
        "\n===== GENERATED REPLY =====\n"
    )

    print(
        result["reply"]
    )


    print(
        f"\nGeneration mode: "
        f"{result['mode']}"
    )


    print(
        "\n===== HISTORICAL EVIDENCE =====\n"
    )


    if (
        evidence is None
        or evidence.empty
    ):

        print(
            "No relevant historical evidence found."
        )


    else:

        for i, row in evidence.iterrows():

            print(
                f"--- Evidence {i + 1} ---"
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


            print()