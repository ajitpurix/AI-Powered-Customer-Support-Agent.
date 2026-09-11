import pandas as pd
import numpy as np

from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import MiniBatchKMeans


# ============================================================
# Configuration
# ============================================================

DATA_PATH = Path("data/processed/apple_conversations.csv")

N_SAMPLE = 20000
N_CLUSTERS = 12


# ============================================================
# Load data
# ============================================================

print("Loading AppleSupport conversations...")

df = pd.read_csv(DATA_PATH)

print(f"Total conversations: {len(df):,}")


# ============================================================
# Clean text
# ============================================================

df["customer_text"] = (
    df["customer_text"]
    .fillna("")
    .astype(str)
    .str.replace(r"http\S+", " ", regex=True)
    .str.replace(r"@\w+", " ", regex=True)
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
)

# Remove extremely short messages
df = df[df["customer_text"].str.len() >= 10].copy()

print(f"Usable messages: {len(df):,}")


# ============================================================
# Sample data
# ============================================================

sample_size = min(N_SAMPLE, len(df))

sample = df.sample(
    n=sample_size,
    random_state=42
).reset_index(drop=True)

print(f"Using {len(sample):,} messages for discovery...")


# ============================================================
# TF-IDF
# ============================================================

print("\nBuilding TF-IDF representation...")

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=10000,
    ngram_range=(1, 2),
    min_df=5,
    max_df=0.95
)

X = vectorizer.fit_transform(sample["customer_text"])

print(f"TF-IDF matrix: {X.shape}")


# ============================================================
# Clustering
# ============================================================

print("\nRunning clustering...")

model = MiniBatchKMeans(
    n_clusters=N_CLUSTERS,
    random_state=42,
    batch_size=1024,
    n_init=10
)

clusters = model.fit_predict(X)

sample["cluster"] = clusters


# ============================================================
# Display cluster keywords + examples
# ============================================================

terms = np.array(vectorizer.get_feature_names_out())

print("\n")
print("=" * 80)
print("DISCOVERED CUSTOMER SUPPORT TOPICS")
print("=" * 80)


for cluster_id in range(N_CLUSTERS):

    cluster_indices = np.where(clusters == cluster_id)[0]

    # Get important terms for this cluster
    center = model.cluster_centers_[cluster_id]

    top_indices = center.argsort()[-15:][::-1]

    top_terms = terms[top_indices]

    print("\n")
    print("=" * 80)
    print(f"CLUSTER {cluster_id}")
    print("=" * 80)

    print("\nTop terms:")
    print(", ".join(top_terms))

    print("\nExample customer messages:")

    examples = sample.iloc[cluster_indices[:5]]

    for _, row in examples.iterrows():
        print(f"- {row['customer_text'][:300]}")


# ============================================================
# Save clustered data
# ============================================================

output_path = Path(
    "data/processed/apple_intent_discovery.csv"
)

sample.to_csv(
    output_path,
    index=False
)

print("\n")
print(f"Saved discovery data to: {output_path}")