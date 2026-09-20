import re

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import joblib

from config import (
    PROCESSED_FILE,
    MODEL_FILE,
    VECTORIZER_FILE,
    FIGURE_DIR,
    PROCESSED_DIR,
    COLORS
)


# ============================================================
# TOPICS
# ============================================================

TOPICS = {

    "Artificial Intelligence": [
        "ai",
        "artificial intelligence",
        "machine learning",
        "chatgpt",
        "openai"
    ],

    "Technology": [
        "technology",
        "software",
        "computer",
        "smartphone",
        "internet"
    ],

    "Sports": [
        "football",
        "soccer",
        "cricket",
        "basketball",
        "tennis"
    ],

    "Climate": [
        "climate",
        "climate change",
        "global warming",
        "carbon",
        "environment"
    ],

    "Education": [
        "education",
        "school",
        "university",
        "student",
        "teacher"
    ]
}


def create_topic_regex(keywords):

    return "|".join(
        re.escape(k)
        for k in keywords
    )


def classify_topics(df):

    model = joblib.load(
        MODEL_FILE
    )

    vectorizer = joblib.load(
        VECTORIZER_FILE
    )

    df["prediction"] = model.predict(
        vectorizer.transform(
            df["clean_text"].astype(str)
        )
    )

    df["predicted_sentiment"] = df[
        "prediction"
    ].map({
        0: "Negative",
        1: "Positive"
    })

    return df


def topic_analysis(df):

    records = []

    for topic, keywords in TOPICS.items():

        pattern = create_topic_regex(
            keywords
        )

        mask = df["clean_text"].str.contains(
            pattern,
            case=False,
            regex=True,
            na=False
        )

        topic_df = df[mask].copy()

        if len(topic_df) == 0:
            continue

        sentiment_counts = (
            topic_df["predicted_sentiment"]
            .value_counts()
        )

        total = len(topic_df)

        positive = sentiment_counts.get(
            "Positive",
            0
        )

        negative = sentiment_counts.get(
            "Negative",
            0
        )

        positive_pct = (
            positive / total * 100
        )

        negative_pct = (
            negative / total * 100
        )

        records.append({
            "topic": topic,
            "tweets": total,
            "positive": positive,
            "negative": negative,
            "positive_pct": positive_pct,
            "negative_pct": negative_pct
        })

    return pd.DataFrame(records)


def topic_bar_chart(results):

    plot_df = results.melt(
        id_vars=["topic"],
        value_vars=[
            "positive_pct",
            "negative_pct"
        ],
        var_name="sentiment",
        value_name="percentage"
    )

    plot_df["sentiment"] = (
        plot_df["sentiment"]
        .map({
            "positive_pct": "Positive",
            "negative_pct": "Negative"
        })
    )

    plt.figure(
        figsize=(14, 8)
    )

    sns.barplot(
        data=plot_df,
        x="topic",
        y="percentage",
        hue="sentiment",
        palette={
            "Positive": COLORS["positive"],
            "Negative": COLORS["negative"]
        }
    )

    plt.title(
        "Public Sentiment by Topic",
        loc="left"
    )

    plt.xlabel("Topic")
    plt.ylabel("Sentiment Percentage (%)")

    plt.xticks(
        rotation=25,
        ha="right"
    )

    plt.legend(
        title="Sentiment"
    )

    sns.despine()

    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / "09_topic_sentiment.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


def topic_heatmap(results):

    heatmap_df = results.set_index(
        "topic"
    )[
        [
            "positive_pct",
            "negative_pct"
        ]
    ]

    heatmap_df.columns = [
        "Positive",
        "Negative"
    ]

    plt.figure(
        figsize=(10, 7)
    )

    sns.heatmap(
        heatmap_df,
        annot=True,
        fmt=".1f",
        cmap="RdYlGn",
        center=50,
        linewidths=1,
        linecolor="white"
    )

    plt.title(
        "Topic Sentiment Heatmap",
        loc="left"
    )

    plt.xlabel("Sentiment")
    plt.ylabel("Topic")

    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / "10_topic_sentiment_heatmap.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


def main():

    df = pd.read_csv(
        PROCESSED_FILE
    )

    # For computational efficiency,
    # change this to a sample for exploratory analysis
    # if necessary.
    df = classify_topics(df)

    results = topic_analysis(
        df
    )

    print("\nTOPIC SENTIMENT RESULTS")
    print("=" * 70)
    print(results)

    results.to_csv(
        PROCESSED_DIR / "topic_predictions.csv",
        index=False
    )

    topic_bar_chart(
        results
    )

    topic_heatmap(
        results
    )

    print("\nTopic analysis completed.")


if __name__ == "__main__":
    main()