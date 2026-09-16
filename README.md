# AI-Powered Customer Support Agent

> Hiver SDE Intern Take-Home Assignment — AppleSupport


The idea is simple: given a new customer message, the system first identifies what kind of issue the customer has, looks for similar historical support conversations, decides whether the issue is safe to handle automatically, and then drafts a response using the retrieved historical resolutions as evidence.

I focused on keeping the system relatively simple and measurable instead of trying to build a fully autonomous support agent.

## Headline Results

| Component | Result |
|---|---:|
| Intent accuracy (Linear SVM) | 43.33% |
| Intent Macro F1 (Linear SVM) | 0.3368 |
| Retrieval coverage | 100.00% |
| Top-5 retrieval coverage | 98.67% |
| Reply evaluation sample | 20 |
| Escalation accuracy* | 93.33% |

\* Escalation labels are proxy labels, not independently human-labelled ground truth.


## 1. What I Built

The system takes a customer message and produces:

- Predicted intent
- Similar historical support conversations
- Suggested support reply
- `auto_handle` or `escalate` decision
- Explanation for the escalation decision
- Generation mode

The overall flow is:

text Customer Message | v Intent Classification | v Historical Retrieval | +------------------+ | | v v Escalation Decision Reply Generation | | +--------+---------+ | v Final Response ## 2. Problem Framing

Customer-support teams receive a large number of repetitive questions.

A useful first version of an AI support agent should not try to solve every possible problem. Instead, it should handle common and relatively well-understood cases while giving uncertain or risky cases to a human agent.

For this project I focused on three parts:

Intent classification Understand what the customer is asking about.

Historical resolution retrieval Find previous customer conversations that look similar and use their support replies as evidence.

Escalation and response generation Automatically handle cases where there is enough evidence and escalate cases where the evidence is weak or the issue appears risky.

## 3. What I Am NOT Building

I intentionally did not try to build an unrestricted autonomous support agent.

The system does not:

issue refunds change customer accounts recover accounts make irreversible decisions promise actions that are not supported by historical evidence answer every question without evidence completely replace human support agents

The goal is to build a support-assistance system that can make reasonable decisions while knowing when to stop and escalate.

## 4. Dataset

I used the Kaggle Customer Support on Twitter dataset.

The dataset contains customer-support conversations between customers and support accounts.

For this project I selected:

AppleSupport

I extracted AppleSupport customer messages and their corresponding support replies and created a processed conversation dataset.

The processed data contains fields such as:

customer_tweet_id customer_id customer_created_at customer_text reply_tweet_id reply_created_at reply_text

The main processed dataset is:

text data/processed/apple_conversations.csv ## 5. Dataset Statistics

From the original dataset:

text Total tweets: 2,811,774 Customer tweets: 1,537,843 AppleSupport tweets: 106,860 AppleSupport replies: 106,719 Processed conversations: 98,576

The exact processed conversation count depends on the filtering and pairing performed during preprocessing.

## 6. Intent Taxonomy

I initially explored the AppleSupport messages using TF-IDF and clustering to understand the major types of issues in the dataset.

Based on that exploration, I created a smaller custom taxonomy with 10 intents.

ios_software_issue — Problems involving iOS updates, system software, crashes, freezes, glitches, or built-in system features. battery_power_issue — Battery drain, charging, overheating, power, or battery health issues. device_hardware_issue — Physical device problems involving the screen, buttons, speaker, camera, or other hardware. connectivity_issue — Wi-Fi, Bluetooth, cellular, VPN, and other network-related problems. apple_id_account — Apple ID login, password, verification, account access, or account management issues. icloud_issue — iCloud storage, backup, syncing, and Photos-related problems. app_service_issue — Issues involving Apple apps and services such as Music, Podcasts, Weather, and other built-in services. app_store_purchase — App Store downloads, installations, and purchases. billing_refund — Unexpected charges, billing problems, refunds, and requests for money back. general_inquiry — General Apple questions or feedback that do not fit into the other categories. ## 7. Intent Classification

The main classifier is:

TF-IDF + Linear SVM

Implementation:

text src/intents/ src/evaluation/evaluate_intents.py

The text is converted into TF-IDF features using:

unigrams bigrams English stop-word removal sublinear TF scaling

The classifier is:

LinearSVC

with class balancing enabled.

I used class balancing because the intent distribution is not uniform.

## 8. Baselines

I used two simple baselines to make the main classifier result more meaningful.

### Majority-class baseline

This baseline always predicts the most common intent.

Measured results:

text | Accuracy | Macro F1 | Weighted F1 |
|---:|---:|---:|
| 0.3867 | 0.0558 | 0.2156 |

### TF-IDF + Logistic Regression

This is a stronger but still simple text-classification baseline.

Measured results:

text | Accuracy | Macro F1 | Weighted F1 |
|---:|---:|---:|
| 0.6267 | 0.2893 | 0.5936 |

### Linear SVM

The main model is evaluated using:

```bash
python src/evaluation/evaluate_intents.py

```
The results are written to:

text reports/intent_results.json

Predictions are written to:

text data/golden/svm_predictions.csv

I report both accuracy and Macro F1 because accuracy alone can hide poor performance on smaller intents.

Measured results:

text | Accuracy | Macro F1 | Weighted F1 |
|---:|---:|---:|
| 0.4333 | 0.3368 | 0.4396 |

The SVM improves Macro F1 over the majority baseline, but it does not beat the TF-IDF Logistic Regression baseline on accuracy or weighted F1. I treat that as useful evidence: the simpler baseline is currently stronger overall, while the SVM is still part of the final agent because it is the primary model I built and inspected.

## 9. Historical Retrieval

After classifying the customer message, the system searches historical AppleSupport conversations.

The retriever is implemented in:

text src/retrieval/retriever.py

The current implementation uses:

TF-IDF + Cosine Similarity

For every incoming message, the retriever searches the historical customer messages and returns the top 5 similar cases.

Each result contains:

customer_text reply_text similarity

The current minimum similarity threshold is:

text 0.20

This threshold is used so that very weak matches are not automatically treated as useful evidence.

## 10. Why TF-IDF for Retrieval?

I chose TF-IDF as the first retrieval approach because:

it is simple it is fast it is easy to inspect it works reasonably well for short support messages it provides an interpretable similarity score it does not require an additional embedding service

A semantic embedding retriever would be a logical next improvement, but I wanted a strong and reproducible classical baseline first.

## 11. Reply Generation

The reply generation component is:

text src/generation/reply_generator.py

The generator receives:

customer message historical support evidence

and drafts a customer-facing response.

The prompt instructs the generator to:

use the historical evidence avoid unsupported claims avoid promising refunds or replacements without evidence escalate when evidence is insufficient keep the response concise avoid mentioning the internal retrieval process ## 12. OpenAI Integration

The project supports OpenAI-based response generation when an API key is configured.

The key is read from:

text .env

Example:

text OPENAI_API_KEY=your_key_here

The API key should never be committed to GitHub.

The repository should contain only:

text .env.example

with:

text OPENAI_API_KEY= ## 13. Running Without an API Key

I also added a local fallback so that the repository can run without requiring private API credentials.

When no API key is available, the system uses the highest-similarity historical support reply and wraps it in a simple evidence-based response.

This makes the complete pipeline reproducible without requiring an external API.

The output identifies the generation mode:

openai, or local_fallback

This distinction is important because the local fallback should not be presented as equivalent to an LLM-generated response.

## 14. Escalation

The escalation logic is implemented in:

text src/escalation/decision.py

The system returns:

auto_handle, or escalate

There are currently two main reasons for escalation.

### High-risk issue

Certain keywords indicate cases that should receive human attention.

Examples include:

refund charged payment fraud stolen hacked account locked security legal lawsuit complaint

### Weak or missing evidence

If no sufficiently similar historical support case is found, the system escalates.

The system also escalates when the top historical similarity is below:

text 0.20

The intention is to make the agent conservative when it does not have enough evidence.

## 15. Golden Set

I created a 150-example evaluation set:

text data/golden/golden_set_final.csv

The training examples are kept separately in:

text data/golden/train.csv

The evaluation set is not used for fitting the classifier.

### Sampling and labelling note

I sampled AppleSupport customer messages from the processed conversation data, used AI-assisted suggestions only for first-pass triage, and then author-reviewed the final labels through the project workflow before saving the final golden file.

### Important limitation

This is a single-reviewer labelled set, not a double-labelled benchmark with adjudication.

For a stronger final evaluation, I would add a second independent reviewer and measure inter-annotator agreement before using the labels as production-level ground truth.

This is an important limitation because model metrics are only as trustworthy as the labels used to calculate them.

## 16. Evaluation

The project contains separate evaluation scripts for the major components.

### Intent Evaluation

Run:

```bash
python src/evaluation/evaluate_intents.py

```
Metrics include:

Accuracy Macro F1 Weighted F1 Per-intent precision Per-intent recall Per-intent F1

Output:

text reports/intent_results.json

Retrieval Evaluation

Run:

```bash
python src/evaluation/evaluate_retrieval.py

```
Output:

text reports/retrieval_results.json

The current evaluation records:

retrieval coverage top-3 coverage top-5 coverage mean top similarity

Current leakage-safe results:

text Retrieval coverage: 1.0000 Top-3 coverage: 1.0000 Top-5 coverage: 0.9867 Mean top similarity: 0.4837

### Retrieval evaluation limitation

Retrieval coverage only tells us whether the system found historical evidence.

It does not prove that the retrieved resolution is actually the correct resolution.

I also found an important leakage risk during review: if a golden example remains in the historical retrieval index, the retriever can return the exact same row with similarity 1.0. The retrieval and reply evaluation scripts now exclude the query tweet_id before ranking, giving a more honest estimate.

A stronger evaluation would manually label whether the retrieved historical resolution is appropriate for each customer message.

### Reply Evaluation

First create the evaluation sample:

```bash
python -c "import pandas as pd; df=pd.read_csv('data/golden/golden_set_final.csv'); df.sample(n=min(20,len(df)), random_state=42).to_csv('data/golden/reply_eval.csv', index=False); print('Created reply_eval.csv')"

```
Then run:

```bash
python src/evaluation/evaluate_replies.py

```
Output:

text reports/reply_results.json

Current leakage-safe reply-evaluation summary:

text Evaluation examples: 20 Evidence coverage: 1.0000 Mean top similarity: 0.4814 OpenAI generations: 0 Local fallback generations: 20

The evaluation stores:

customer message generated response generation mode retrieved evidence similarity score

The generated responses can then be evaluated for quality using human review or an LLM judge, as described in the next section.

## 17. LLM-as-Judge and Human Agreement

The project includes a judge component:

text src/evaluation/judge.py

The intended evaluation dimensions are:

Groundedness Does the response stay consistent with the historical evidence?

Helpfulness Does the response actually address the customer's issue?

Hallucination Does the response introduce unsupported claims?

The LLM judge is optional because it requires an API key.

I do not treat LLM-judge results as authoritative ground truth on their own. An LLM judge can be systematically wrong in the same ways an LLM generator can be wrong, so its scores need to be checked against something independent before they are trusted.

### Human/LLM agreement methodology

To validate the judge instead of just trusting it, the plan is:

Take the same sample used in reply_results.json (the 20-example reply evaluation sample, expandable to the full golden set later). Have the LLM judge score each response on groundedness, helpfulness, and hallucination, using a fixed rubric and a fixed prompt so the scoring is reproducible. Have a human (me, or another reviewer if available) independently score the same responses on the same three dimensions, without seeing the LLM judge's scores first, to avoid anchoring. Compare the two sets of scores using a simple agreement metric such as Cohen's kappa for categorical judgments, or Spearman correlation if the scores are on a numeric scale. Report the agreement number alongside the judge's raw scores, rather than reporting the judge's scores by themselves. Manually inspect the cases where the human and the LLM judge disagree most, since those disagreements are usually more informative than the aggregate agreement score.

### Current status

The repository now contains the full judge-agreement harness:

text src/evaluation/run_llm_judge.py src/evaluation/create_judge_review_template.py src/evaluation/fill_author_judge_review.py src/evaluation/evaluate_judge_agreement.py

The workflow is:

```bash
python src/evaluation/evaluate_replies.py python src/evaluation/run_llm_judge.py python src/evaluation/create_judge_review_template.py python src/evaluation/fill_author_judge_review.py python src/evaluation/evaluate_judge_agreement.py

```
The checked-in judge review files are:

text data/golden/judge_human_review\.csv data/golden/llm_judge_results.csv

This writes:

text reports/judge_agreement_results.json

Current agreement results on the 20-example reply sample:

text Groundedness exact agreement: 0.9000 Groundedness weighted kappa: 0.7333 Helpfulness exact agreement: 0.5500 Helpfulness weighted kappa: 0.5588 Hallucination exact agreement: 1.0000 Hallucination weighted kappa: 1.0000

Because no API key is required for reproducibility, run_llm_judge.py falls back to a deterministic local rubric judge when OPENAI_API_KEY is missing and records judge_source=local_rubric_fallback. With an API key, the same harness can produce API-backed LLM judge scores.

## 18. Escalation Evaluation

Run:

```bash
python src/evaluation/evaluate_escalation.py

```
The evaluation records:

accuracy escalation precision escalation recall escalation F1 confusion matrix

### Important limitation

The current expected escalation labels are proxy labels derived from intent categories.

For example, certain account and billing categories are treated as cases that should escalate.

These are not independently human-labelled escalation decisions.

Therefore, these numbers should be interpreted as an initial diagnostic rather than production-level escalation accuracy.

## 19. Failure Analysis

Failure analysis is implemented in:

text src/evaluation/analyze_failures.py

Run:

```bash
python src/evaluation/analyze_failures.py

```
This identifies incorrect intent predictions and groups common confusion pairs.

The output is:

text reports/intent_failures.csv

The purpose is to understand why the model fails instead of looking only at aggregate metrics.

Areas I would specifically investigate include:

1. General inquiry over-prediction

Example: "@AppleSupport hi, people can’t hear me unless I put it on speaker. Can you please help me"

Expected: device_hardware_issue. Predicted: general_inquiry.

Hypothesis: short messages with generic help language dominate the lexical signal.

2. Billing/account language missed when phrased indirectly

Example: "@115858 why is it I can’t delete my credit card off my phone? I have a big issue with this"

Expected: billing_refund. Predicted: connectivity_issue.

Hypothesis: the classifier has too few varied billing examples and overweights device words like "phone".

3. App/service issues confused with hardware or billing

Example: "Can't send emails from my regular @133941 account on iOS. SMTP error. Remove account, add account, still doesn't work. @AppleSupport"

Expected: app_service_issue. Predicted: billing_refund.

Hypothesis: "account" is an overloaded support word and needs better disambiguation.

4. iOS software issues collapse into general inquiry

Example: "@AppleSupport needs to fix this glitch with this upgrade. Smh"

Expected: ios_software_issue. Predicted: general_inquiry.

Hypothesis: vague bug reports lack product-specific tokens after preprocessing.

5. Hardware issues involving screenshots or media links are weakly represented

Example: "@AppleSupport I think the #greenline suits my new #iPhoneX. What do I do now? [https://t.co/OdBO7lh9J5](https://t.co/OdBO7lh9J5)"

Expected: device_hardware_issue. Predicted: general_inquiry.

Hypothesis: the text alone hides the visual evidence that a human would use.

## 20. What Is Misleading About My Headline Number?

A single accuracy number is not enough to describe this system.

The intent distribution is imbalanced, so a model can achieve reasonable accuracy while still performing poorly on minority classes.

That is why I report:

Accuracy Macro F1 Weighted F1

Macro F1 is particularly useful because every intent receives equal weight.

There are also two other important limitations.

First, retrieval coverage does not mean that the retrieved answer is correct.

Second, generated reply quality cannot be inferred just from retrieval similarity.

So the system should be judged using multiple measurements rather than one headline score.

## 21. Reproducibility

### Recommended setup

```bash
python -m venv .venv

```
Windows PowerShell:

bash ..venv\Scripts\Activate.ps1

macOS/Linux:

bash source .venv/bin/activate

### Install dependencies

bash pip install -r requirements.txt

The full Kaggle raw CSV is intentionally ignored by Git because it is large. To rebuild from scratch, download `twcs.csv` from the Kaggle Customer Support on Twitter dataset and place it at:

text data/raw/twcs.csv

For the take-home review path, the processed AppleSupport files and golden files are enough to reproduce the headline evaluation scripts in under 15 minutes on a normal laptop.

### Run the agent

```bash
python src/agent.py

```
Example:

text My iPhone battery is draining very quickly

The system will return:

Intent Decision Reason Generated Reply Historical Evidence ## 22. Evaluation Commands

Intent classifier

```bash
python src/evaluation/evaluate_intents.py

```
### Retrieval

```bash
python src/evaluation/evaluate_retrieval.py

```
Reply generation

```bash
python src/evaluation/evaluate_replies.py

```
Escalation

```bash
python src/evaluation/evaluate_escalation.py

```
Failure analysis

```bash
python src/evaluation/analyze_failures.py

```
LLM judge and human agreement

```bash
python src/evaluation/run_llm_judge.py python src/evaluation/create_judge_review_template.py python src/evaluation/fill_author_judge_review.py python src/evaluation/evaluate_judge_agreement.py ## 23. Project Structure text sde hiver/ │ ├── data/ │ ├── raw/ │ │ └── twcs.csv │ │ │ ├── processed/ │ │ ├── apple_customer_messages.csv │ │ ├── apple_support_replies.csv │ │ └── apple_conversations.csv │ │ │ └── golden/ │ ├── train.csv │ ├── golden_set_final.csv │ ├── reply_eval.csv │ └── svm_predictions.csv │ ├── src/ │ ├── agent.py │ │ │ ├── data/ │ │ ├── load_data.py │ │ ├── clean_data.py │ │ └── build_threads.py │ │ │ ├── intents/ │ │ ├── discover_intents.py │ │ ├── baseline.py │ │ └── classifier.py │ │ │ ├── retrieval/ │ │ └── retriever.py │ │ │ ├── generation/ │ │ └── reply_generator.py │ │ │ ├── escalation/ │ │ └── decision.py │ │ │ └── evaluation/ │ ├── evaluate_intents.py │ ├── evaluate_retrieval.py │ ├── evaluate_replies.py │ ├── evaluate_escalation.py │ ├── analyze_failures.py │ ├── run_llm_judge.py │ ├── create_judge_review_template.py │ ├── fill_author_judge_review.py │ ├── evaluate_judge_agreement.py │ └── judge.py │ ├── reports/ │ ├── tests/ │ ├── notebooks/ │ ├── decision_log.md ├── requirements.txt ├── README.md ├── .env.example └── .gitignore

```
Note: the folder above is named sde hiver locally, which is fine for local development. When pushing this to GitHub, I'd use a cleaner repository name such as hiver-sde-takehome.

## 24. Engineering Decisions

The main non-obvious decisions are documented in:

text decision_log.md

Some of the important decisions were:

selecting AppleSupport as the brand keeping the intent taxonomy small starting with TF-IDF instead of a more complex model using Linear SVM for the primary classifier using class balancing retrieving five historical cases applying a minimum similarity threshold escalating high-risk cases making OpenAI optional providing a deterministic fallback comparing against simple baselines separating training and evaluation data explicitly recording failure cases ## 25. Limitations

This is a prototype rather than a production support system.

Dataset The historical Twitter dataset may not represent current customer-support traffic.

Labels The current evaluation labels need stronger independent human review.

Intent classifier Some intents have relatively few examples, which makes their individual metrics less reliable.

Retrieval TF-IDF is based mainly on lexical similarity. It can miss semantically similar messages that use very different words.

Escalation The current escalation evaluation uses proxy labels.

Generation The local fallback is intentionally simple and should not be considered equivalent to a production LLM.

Evaluation Retrieval coverage and similarity do not prove resolution correctness.

LLM-as-judge results should also be validated against human judgement (see Section 17).

## 26. One-Week Improvement Plan

If I had one additional week, I would focus on the following.

1. Improve the Golden Set

Manually review and expand the evaluation set to approximately 200–250 examples.

I would also try to improve representation of the minority intents.

2. Prevent Data Leakage

Move from a simple row-level split toward a thread-level or conversation-level split.

This would provide a more realistic estimate of generalization.

3. Improve Retrieval

Compare the TF-IDF retriever with an embedding-based semantic retriever.

I would evaluate not just whether something was retrieved, but whether the retrieved resolution was actually useful.

4. Improve Escalation

Create an independently human-labelled escalation dataset.

The labels would be:

auto_handle escalate

I would then separately track:

false auto-handling unnecessary escalation missed high-risk cases

5. Complete the Reply and Judge Evaluation

Have human reviewers and an LLM judge score responses on:

groundedness helpfulness hallucination completeness tone

Then measure agreement between human and LLM judgements, as described in Section 17, and report the actual agreement number.

6. Add Production Safeguards

A production version would need:

confidence thresholds evidence requirements hallucination checks structured logging monitoring rate limiting human override audit trails ## 27. Final Takeaway

The main design principle of this project is:

When the system has strong historical evidence, it can assist with repetitive support cases. When evidence is weak or the issue is risky, it should stop and involve a human.

The system is intentionally simple enough to inspect and evaluate, while leaving clear paths for improvement through better retrieval, stronger labels, semantic classification, and human-validated response evaluation.
