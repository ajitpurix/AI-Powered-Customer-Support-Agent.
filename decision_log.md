# Decision Log

| # | Decision | Why |
|---|---|---|
| 1 | Selected AppleSupport | Large number of support interactions |
| 2 | Used 10 custom intents | Small enough to evaluate and broad enough to cover major issue types |
| 3 | Used TF-IDF features | Fast, interpretable baseline for short support messages |
| 4 | Used Linear SVM | Strong classical classifier for sparse text |
| 5 | Used class balancing | Intent distribution is highly imbalanced |
| 6 | Used historical customer messages for retrieval | Directly connects new issues with previous resolutions |
| 7 | Used cosine similarity | Simple and interpretable semantic matching |
| 8 | Retrieved top 5 cases | Provides multiple evidence candidates without excessive context |
| 9 | Added similarity threshold | Prevents weak historical matches from being treated as reliable evidence |
| 10 | Added escalation rules | High-risk cases should receive human review |
| 11 | Made OpenAI optional | Repository can run without exposing credentials |
| 12 | Added local fallback generation | Keeps the complete pipeline reproducible without API access |
| 13 | Evaluated against majority baseline | Establishes minimum performance |
| 14 | Evaluated against Logistic Regression | Provides a stronger classical baseline |
| 15 | Added failure analysis | Identifies where the classifier needs improvement |