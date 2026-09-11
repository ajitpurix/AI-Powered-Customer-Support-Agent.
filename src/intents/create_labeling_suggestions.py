import pandas as pd
from pathlib import Path
import re

INPUT_PATH = Path("data/golden/labeling_candidates.csv")
OUTPUT_PATH = Path("data/golden/labeling_suggestions.csv")


def suggest_intent(text):
    text = str(text).lower()

    rules = [
        (
            "battery_power_issue",
            [
                "battery", "charging", "charge", "charger",
                "overheat", "overheating", "power", "drain"
            ],
        ),
        (
            "connectivity_issue",
            [
                "wifi", "wi-fi", "bluetooth", "cellular",
                "mobile data", "network", "vpn", "signal"
            ],
        ),
        (
            "apple_id_account",
            [
                "apple id", "appleid", "password",
                "sign in", "signin", "login", "log in",
                "verification code", "account"
            ],
        ),
        (
            "icloud_issue",
            [
                "icloud", "i cloud", "backup", "syncing",
                "sync", "icloud storage"
            ],
        ),
        (
            "billing_refund",
            [
                "refund", "charged", "charge", "billing",
                "payment", "money back", "subscription charge"
            ],
        ),
        (
            "app_store_purchase",
            [
                "app store", "download app", "install app",
                "purchase app", "can't download", "cannot download"
            ],
        ),
        (
            "app_service_issue",
            [
                "apple music", "itunes", "podcast",
                "podcasts", "weather app", "weather widget",
                "screen recording", "siri", "music app"
            ],
        ),
        (
            "device_hardware_issue",
            [
                "screen", "display", "speaker", "microphone",
                "camera", "button", "home button", "broken",
                "cracked", "physical damage"
            ],
        ),
        (
            "ios_software_issue",
            [
                "ios", "update", "updated", "software",
                "freeze", "freezing", "crash", "crashing",
                "glitch", "bug", "latest update"
            ],
        ),
    ]

    for intent, keywords in rules:
        for keyword in keywords:
            if keyword in text:
                return intent

    return "general_inquiry"


df = pd.read_csv(INPUT_PATH)

df["suggested_intent"] = df["customer_text"].apply(suggest_intent)

# Human reviewer fills this column.
df["human_intent"] = ""

# Optional confidence indicator based on keyword matching.
df["suggestion_status"] = df["suggested_intent"].apply(
    lambda x: "review_required"
)

output_columns = [
    "customer_tweet_id",
    "customer_text",
    "reply_text",
    "suggested_intent",
    "human_intent",
    "suggestion_status",
]

df[output_columns].to_csv(OUTPUT_PATH, index=False)

print("===== LABELING SUGGESTIONS =====")
print(f"Candidates: {len(df)}")
print(f"Saved to: {OUTPUT_PATH}")

print("\n===== SUGGESTED DISTRIBUTION =====")
print(df["suggested_intent"].value_counts())