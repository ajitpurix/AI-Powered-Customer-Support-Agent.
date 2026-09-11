import pandas as pd
from pathlib import Path

DATA_PATH = Path("data/raw/twcs.csv")

print("Loading dataset...")
df = pd.read_csv(DATA_PATH)

# Convert IDs to strings so they can be matched safely
df["tweet_id"] = df["tweet_id"].astype(str)
df["author_id"] = df["author_id"].astype(str)

# Customer = inbound
# Brand/support = outbound
customer = df[df["inbound"] == True].copy()
support = df[df["inbound"] == False].copy()

print("\n===== BASIC STATS =====")
print(f"Total tweets       : {len(df):,}")
print(f"Customer tweets    : {len(customer):,}")
print(f"Support tweets     : {len(support):,}")

# ---------------------------------------------------------
# Find support accounts
# ---------------------------------------------------------

support_stats = (
    support.groupby("author_id")
    .agg(
        support_tweets=("tweet_id", "count"),
        replies=("in_response_to_tweet_id", lambda x: x.notna().sum())
    )
    .reset_index()
)

# Count how many customer tweets received a response
customer_response_count = (
    customer["response_tweet_id"]
    .notna()
)

# Display accounts with substantial support activity
support_stats = support_stats.sort_values(
    "support_tweets",
    ascending=False
)

print("\n===== TOP SUPPORT ACCOUNTS =====")
print(
    support_stats.head(30).to_string(index=False)
)

# Save results
output_path = Path("data/processed/support_accounts.csv")
output_path.parent.mkdir(parents=True, exist_ok=True)

support_stats.to_csv(output_path, index=False)

print(f"\nSaved results to: {output_path}")