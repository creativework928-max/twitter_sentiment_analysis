import re
import html
import pandas as pd
import numpy as np

from config import (
    RAW_FILE,
    PROCESSED_FILE,
    TRAINING_SAMPLE_SIZE,
    RANDOM_STATE
)


COLUMN_NAMES = [
    "target",
    "tweet_id",
    "date",
    "query",
    "user",
    "text"
]


def clean_tweet(text):

    if pd.isna(text):
        return ""

    text = str(text)

    # Decode HTML entities
    text = html.unescape(text)

    # Remove RT marker
    text = re.sub(r"^\s*RT\s+", "", text)

    # Replace URLs with token
    text = re.sub(
        r"https?://\S+|www\.\S+",
        " URL ",
        text
    )

    # Replace mentions
    text = re.sub(
        r"@\w+",
        " USER ",
        text
    )

    # Preserve hashtag word but remove '#'
    text = re.sub(
        r"#(\w+)",
        r"\1",
        text
    )

    # Normalize repeated characters:
    # cooooool -> coool
    text = re.sub(
        r"(.)\1{3,}",
        r"\1\1\1",
        text
    )

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    # Lowercase
    text = text.lower().strip()

    return text


def load_data():

    df = pd.read_csv(
        RAW_FILE,
        encoding="latin-1",
        header=None,
        names=COLUMN_NAMES
    )

    return df


def preprocess_dataset(df):

    # Keep binary sentiment labels
    df = df[df["target"].isin([0, 4])].copy()

    # Remove missing tweets
    df = df.dropna(subset=["text"])

    # Remove duplicate tweets
    df = df.drop_duplicates(subset=["text"])

    # Clean text
    df["clean_text"] = df["text"].apply(clean_tweet)

    # Remove empty records
    df = df[df["clean_text"].str.len() > 0]

    # Convert labels
    df["sentiment"] = df["target"].map({
        0: "Negative",
        4: "Positive"
    })

    # Binary numerical label
    df["label"] = df["target"].map({
        0: 0,
        4: 1
    })

    # Date
    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    )

    # Tweet length
    df["tweet_length"] = df["clean_text"].str.len()

    # Word count
    df["word_count"] = (
        df["clean_text"]
        .str.split()
        .str.len()
    )

    if TRAINING_SAMPLE_SIZE is not None:

        df = (
            df.groupby("label", group_keys=False)
            .apply(
                lambda x: x.sample(
                    min(
                        len(x),
                        TRAINING_SAMPLE_SIZE // 2
                    ),
                    random_state=RANDOM_STATE
                ),
                include_groups=False
            )
            .reset_index(drop=True)
        )

    return df


def main():

    print("Loading data...")

    df = load_data()

    print(f"Original rows: {len(df):,}")

    df = preprocess_dataset(df)

    print(f"Processed rows: {len(df):,}")

    df.to_csv(
        PROCESSED_FILE,
        index=False,
        encoding="utf-8"
    )

    print(f"\nSaved processed dataset:")
    print(PROCESSED_FILE)

    print("\nSentiment distribution:")
    print(df["sentiment"].value_counts())


if __name__ == "__main__":
    main()