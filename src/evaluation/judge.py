from pathlib import Path
import sys
import os
import json

from dotenv import load_dotenv
from openai import OpenAI

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(BASE_DIR))

load_dotenv()


class ReplyJudge:

    def __init__(self):

        api_key = os.getenv("OPENAI_API_KEY")

        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY not found in .env"
            )

        self.client = OpenAI(
            api_key=api_key
        )

    def build_prompt(
        self,
        customer_message,
        historical_evidence,
        generated_reply
    ):

        evidence_text = ""

        for i, row in historical_evidence.iterrows():

            evidence_text += f"""
Evidence {i + 1}:

Customer:
{row["customer_text"]}

Historical resolution:
{row["reply_text"]}
"""

        prompt = f"""
You are evaluating an AI customer-support reply.

CUSTOMER MESSAGE:
{customer_message}

HISTORICAL SUPPORT EVIDENCE:
{evidence_text}

GENERATED REPLY:
{generated_reply}

Evaluate the generated reply using the following criteria.

1. GROUNDEDNESS
Is the reply supported by the historical evidence?

2. HELPFULNESS
Does the reply address the customer's actual problem?

3. HALLUCINATION
Does the reply introduce claims, policies, actions,
or promises that are not supported by the evidence?

Give each score from 1 to 5.

Scoring:

Groundedness:
1 = completely unsupported
2 = mostly unsupported
3 = partially supported
4 = mostly supported
5 = fully supported

Helpfulness:
1 = does not address the issue
2 = barely useful
3 = somewhat useful
4 = helpful
5 = directly and effectively addresses the issue

Hallucination:
1 = many unsupported claims
2 = several unsupported claims
3 = some unsupported claims
4 = very few unsupported claims
5 = no meaningful unsupported claims

Return ONLY valid JSON:

{{
    "groundedness": 1,
    "helpfulness": 1,
    "hallucination": 1,
    "reason": "short explanation"
}}
"""

        return prompt

    def judge(
        self,
        customer_message,
        historical_evidence,
        generated_reply
    ):

        prompt = self.build_prompt(
            customer_message,
            historical_evidence,
            generated_reply
        )

        response = self.client.responses.create(
            model="gpt-5.5",
            input=prompt
        )

        text = response.output_text.strip()

        try:

            result = json.loads(text)

        except json.JSONDecodeError:

            result = {
                "groundedness": None,
                "helpfulness": None,
                "hallucination": None,
                "reason": text
            }

        return result