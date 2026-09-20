import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import joblib

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
    roc_auc_score
)

from config import (
    PROCESSED_FILE,
    MODEL_FILE,
    VECTORIZER_FILE,
    FIGURE_DIR,
    REPORT_DIR,
    COLORS,
    RANDOM_STATE,
    TEST_SIZE
)


def main():

    df = pd.read_csv(
        PROCESSED_FILE
    )

    X = df["clean_text"].astype(str)
    y = df["label"]

    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    model = joblib.load(
        MODEL_FILE
    )

    vectorizer = joblib.load(
        VECTORIZER_FILE
    )

    X_test_vec = vectorizer.transform(
        X_test
    )

    predictions = model.predict(
        X_test_vec
    )

    # ========================================================
    # CLASSIFICATION REPORT
    # ========================================================

    report = classification_report(
        y_test,
        predictions,
        target_names=[
            "Negative",
            "Positive"
        ],
        digits=4
    )

    print(report)

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        REPORT_DIR / "classification_report.txt",
        "w",
        encoding="utf-8"
    ) as f:

        f.write(report)

    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    cm = confusion_matrix(
        y_test,
        predictions
    )

    plt.figure(
        figsize=(9, 7)
    )

    sns.heatmap(
        cm,
        annot=True,
        fmt=",d",
        cmap="Blues",
        linewidths=1,
        linecolor="white",
        xticklabels=[
            "Negative",
            "Positive"
        ],
        yticklabels=[
            "Negative",
            "Positive"
        ]
    )

    plt.title(
        "Sentiment Classification — Confusion Matrix",
        loc="left"
    )

    plt.xlabel("Predicted Sentiment")
    plt.ylabel("Actual Sentiment")

    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / "07_confusion_matrix.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # ========================================================
    # ROC CURVE
    # ========================================================

    if hasattr(model, "decision_function"):

        scores = model.decision_function(
            X_test_vec
        )

        fpr, tpr, _ = roc_curve(
            y_test,
            scores
        )

        roc_auc = auc(
            fpr,
            tpr
        )

        plt.figure(
            figsize=(10, 7)
        )

        plt.plot(
            fpr,
            tpr,
            color=COLORS["primary"],
            linewidth=3,
            label=f"AUC = {roc_auc:.4f}"
        )

        plt.plot(
            [0, 1],
            [0, 1],
            "--",
            color="#999999"
        )

        plt.xlabel(
            "False Positive Rate"
        )

        plt.ylabel(
            "True Positive Rate"
        )

        plt.title(
            "ROC Curve",
            loc="left"
        )

        plt.legend()

        sns.despine()

        plt.tight_layout()

        plt.savefig(
            FIGURE_DIR / "08_roc_curve.png",
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

        print(
            f"ROC-AUC: {roc_auc:.4f}"
        )


if __name__ == "__main__":
    main()