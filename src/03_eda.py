import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from wordcloud import WordCloud

from config import (
    PROCESSED_FILE,
    FIGURE_DIR,
    COLORS
)


# ============================================================
# PROFESSIONAL STYLE
# ============================================================

sns.set_theme(
    style="whitegrid",
    context="notebook"
)

plt.rcParams.update({
    "figure.figsize": (12, 7),
    "figure.dpi": 150,
    "axes.titlesize": 18,
    "axes.titleweight": "bold",
    "axes.labelsize": 12,
    "axes.labelcolor": "#264653",
    "xtick.color": "#264653",
    "ytick.color": "#264653",
    "font.family": "DejaVu Sans"
})


def save_plot(filename):

    path = FIGURE_DIR / filename

    plt.tight_layout()

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
        facecolor="white"
    )

    plt.close()

    print(f"Saved: {path}")


def load_data():

    df = pd.read_csv(
        PROCESSED_FILE
    )

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce",
        format="mixed"
    )

    return df


# ============================================================
# 1. CLASS DISTRIBUTION
# ============================================================

def class_distribution(df):

    counts = df["sentiment"].value_counts()

    plt.figure(figsize=(10, 6))

    colors = [
        COLORS["negative"],
        COLORS["positive"]
    ]

    ax = sns.barplot(
        x=counts.index,
        y=counts.values,
        hue=counts.index,
        palette=colors,
        legend=False
    )

    plt.title(
        "Twitter Sentiment Distribution",
        loc="left"
    )

    plt.xlabel("Sentiment")
    plt.ylabel("Number of Tweets")

    for container in ax.containers:
        ax.bar_label(
            container,
            fmt="%d",
            padding=4
        )

    sns.despine()

    save_plot("01_class_distribution.png")


# ============================================================
# 2. TWEET LENGTH
# ============================================================

def tweet_length_distribution(df):

    plt.figure(figsize=(12, 7))

    sns.histplot(
        data=df,
        x="tweet_length",
        hue="sentiment",
        bins=50,
        kde=True,
        palette=[
            COLORS["negative"],
            COLORS["positive"]
        ],
        element="step"
    )

    plt.title(
        "Distribution of Tweet Length",
        loc="left"
    )

    plt.xlabel("Tweet Length (Characters)")
    plt.ylabel("Frequency")

    sns.despine()

    save_plot("02_tweet_length_distribution.png")


# ============================================================
# 3. WORD COUNT
# ============================================================

def word_count_distribution(df):

    plt.figure(figsize=(12, 7))

    sns.histplot(
        data=df,
        x="word_count",
        hue="sentiment",
        bins=40,
        kde=True,
        palette=[
            COLORS["negative"],
            COLORS["positive"]
        ],
        element="step"
    )

    plt.title(
        "Tweet Word Count Distribution",
        loc="left"
    )

    plt.xlabel("Number of Words")
    plt.ylabel("Frequency")

    sns.despine()

    save_plot("03_word_count_distribution.png")


# ============================================================
# 4. WORDCLOUD
# ============================================================

def create_wordcloud(text, filename, title, color):

    wc = WordCloud(
        width=1600,
        height=900,
        background_color="white",
        colormap="viridis",
        max_words=150,
        min_font_size=10,
        collocations=False
    ).generate(text)

    plt.figure(figsize=(16, 9))

    plt.imshow(wc, interpolation="bilinear")

    plt.axis("off")

    plt.title(
        title,
        fontsize=22,
        fontweight="bold",
        color=color,
        loc="left"
    )

    save_plot(filename)


def wordclouds(df):

    positive_text = " ".join(
        df.loc[
            df["sentiment"] == "Positive",
            "clean_text"
        ].astype(str)
    )

    negative_text = " ".join(
        df.loc[
            df["sentiment"] == "Negative",
            "clean_text"
        ].astype(str)
    )

    create_wordcloud(
        positive_text,
        "04_positive_wordcloud.png",
        "Positive Tweets — Most Frequent Terms",
        COLORS["positive"]
    )

    create_wordcloud(
        negative_text,
        "05_negative_wordcloud.png",
        "Negative Tweets — Most Frequent Terms",
        COLORS["negative"]
    )


# ============================================================
# 5. SENTIMENT OVER TIME
# ============================================================

def sentiment_over_time(df):

    temp = df.copy()

    # Convert date column to datetime
    temp["date"] = pd.to_datetime(
        temp["date"],
        errors="coerce",
        format="mixed"
    )

    # Remove rows with invalid/missing dates
    temp = temp.dropna(
        subset=["date"]
    )

    print(
        f"Valid dates for time analysis: {len(temp):,}"
    )

    temp["month"] = (
        temp["date"]
        .dt.to_period("M")
        .astype(str)
    )

    monthly = (
        temp.groupby(
            ["month", "sentiment"]
        )
        .size()
        .reset_index(name="count")
    )

    plt.figure(figsize=(15, 7))

    sns.lineplot(
        data=monthly,
        x="month",
        y="count",
        hue="sentiment",
        marker="o",
        palette={
            "Negative": COLORS["negative"],
            "Positive": COLORS["positive"]
        },
        linewidth=2.5
    )

    plt.title(
        "Sentiment Trend Over Time",
        loc="left"
    )

    plt.xlabel("Month")
    plt.ylabel("Tweet Count")

    plt.xticks(rotation=45)

    sns.despine()

    save_plot("06_sentiment_over_time.png")


def main():

    df = load_data()

    print(f"Dataset rows: {len(df):,}")

    class_distribution(df)

    tweet_length_distribution(df)

    word_count_distribution(df)

    wordclouds(df)

    sentiment_over_time(df)

    print("\nEDA completed.")


if __name__ == "__main__":
    main()




























#     import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# from config import REPORT_DIR, FIGURE_DIR, COLORS


# def model_comparison_chart():

#     df = pd.read_csv(
#         REPORT_DIR / "model_comparison.csv"
#     )

#     plot_df = df.melt(
#         id_vars=["model"],
#         value_vars=[
#             "accuracy",
#             "precision",
#             "recall",
#             "f1"
#         ],
#         var_name="metric",
#         value_name="score"
#     )

#     plt.figure(
#         figsize=(13, 7)
#     )

#     sns.barplot(
#         data=plot_df,
#         x="model",
#         y="score",
#         hue="metric",
#         palette="viridis"
#     )

#     plt.ylim(
#         0,
#         1
#     )

#     plt.title(
#         "Machine Learning Model Performance Comparison",
#         loc="left",
#         fontsize=18,
#         fontweight="bold"
#     )

#     plt.xlabel("Model")
#     plt.ylabel("Score")

#     plt.xticks(
#         rotation=15
#     )

#     plt.legend(
#         title="Metric"
#     )

#     sns.despine()

#     plt.tight_layout()

#     plt.savefig(
#         FIGURE_DIR /
#         "11_model_comparison.png",
#         dpi=300,
#         bbox_inches="tight"
#     )

#     plt.close()


# if __name__ == "__main__":
#     model_comparison_chart()
