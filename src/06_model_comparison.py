import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from config import (
    REPORT_DIR,
    FIGURE_DIR,
    COLORS
)


# ============================================================
# MODEL COMPARISON CHART
# ============================================================

def model_comparison_chart():

    # Load model comparison results
    df = pd.read_csv(
        REPORT_DIR / "model_comparison.csv"
    )

    # Convert wide format to long format
    plot_df = df.melt(
        id_vars=["model"],
        value_vars=[
            "accuracy",
            "precision",
            "recall",
            "f1"
        ],
        var_name="metric",
        value_name="score"
    )

    # Create figure
    plt.figure(
        figsize=(13, 7)
    )

    # Plot model performance
    sns.barplot(
        data=plot_df,
        x="model",
        y="score",
        hue="metric",
        palette="viridis"
    )

    # Scores are between 0 and 1
    plt.ylim(
        0,
        1
    )

    plt.title(
        "Machine Learning Model Performance Comparison",
        loc="left",
        fontsize=18,
        fontweight="bold"
    )

    plt.xlabel(
        "Model"
    )

    plt.ylabel(
        "Score"
    )

    plt.xticks(
        rotation=15
    )

    plt.legend(
        title="Metric"
    )

    sns.despine()

    plt.tight_layout()

    # Save figure
    output_path = (
        FIGURE_DIR /
        "11_model_comparison.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
        facecolor="white"
    )

    plt.close()

    print(
        f"Saved: {output_path}"
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    model_comparison_chart()