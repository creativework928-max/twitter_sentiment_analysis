import json
import time

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.pipeline import FeatureUnion

from sklearn.feature_extraction.text import (
    TfidfVectorizer
)

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import ComplementNB

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

import joblib

from config import (
    PROCESSED_FILE,
    MODEL_DIR,
    MODEL_FILE,
    VECTORIZER_FILE,
    METADATA_FILE,
    TEST_SIZE,
    RANDOM_STATE,
    REPORT_DIR
)


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("\nLoading processed dataset...")

    df = pd.read_csv(
        PROCESSED_FILE
    )

    print(
        f"Dataset rows: {len(df):,}"
    )

    required_columns = [
        "clean_text",
        "label"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            "Missing required columns: "
            f"{missing_columns}"
        )

    return df


# ============================================================
# BUILD TF-IDF VECTORIZER
# ============================================================

def build_vectorizer():

    print("\nBuilding TF-IDF vectorizers...")

    # --------------------------------------------------------
    # WORD TF-IDF
    # --------------------------------------------------------

    word_vectorizer = TfidfVectorizer(
        lowercase=False,
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.95,
        sublinear_tf=True,
        max_features=250_000
    )

    # --------------------------------------------------------
    # CHARACTER TF-IDF
    # --------------------------------------------------------

    char_vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        min_df=3,
        sublinear_tf=True,
        max_features=150_000
    )

    # --------------------------------------------------------
    # COMBINE WORD + CHARACTER FEATURES
    # --------------------------------------------------------

    vectorizer = FeatureUnion([
        (
            "word",
            word_vectorizer
        ),
        (
            "char",
            char_vectorizer
        )
    ])

    return vectorizer


# ============================================================
# EVALUATE MODEL
# ============================================================

def evaluate(
    model,
    X_test,
    y_test
):

    predictions = model.predict(
        X_test
    )

    return {
        "accuracy": accuracy_score(
            y_test,
            predictions
        ),

        "precision": precision_score(
            y_test,
            predictions,
            zero_division=0
        ),

        "recall": recall_score(
            y_test,
            predictions,
            zero_division=0
        ),

        "f1": f1_score(
            y_test,
            predictions,
            zero_division=0
        )
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("TWITTER SENTIMENT MODEL TRAINING")
    print("=" * 70)

    # ========================================================
    # LOAD DATA
    # ========================================================

    df = load_data()

    # --------------------------------------------------------
    # Remove missing text/labels
    # --------------------------------------------------------

    df = df[
        ["clean_text", "label"]
    ].dropna()

    df["clean_text"] = (
        df["clean_text"]
        .astype(str)
        .str.strip()
    )

    df = df[
        df["clean_text"] != ""
    ]

    print(
        f"Usable rows: {len(df):,}"
    )

    # ========================================================
    # FEATURES AND TARGET
    # ========================================================

    X = df["clean_text"]

    y = df["label"]

    print("\nClass distribution:")

    print(
        y.value_counts()
    )

    # --------------------------------------------------------
    # Validate binary labels
    # --------------------------------------------------------

    unique_labels = sorted(
        y.unique().tolist()
    )

    print(
        f"\nLabels detected: {unique_labels}"
    )

    if len(unique_labels) != 2:

        raise ValueError(
            "This project requires exactly "
            f"2 sentiment classes. "
            f"Found: {unique_labels}"
        )

    # ========================================================
    # TRAIN / TEST SPLIT
    # ========================================================

    print("\nCreating train/test split...")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print(
        f"Training rows: {len(X_train):,}"
    )

    print(
        f"Testing rows : {len(X_test):,}"
    )

    # ========================================================
    # TF-IDF
    # ========================================================

    print("\n" + "=" * 70)
    print("TF-IDF VECTORIZATION")
    print("=" * 70)

    vectorizer = build_vectorizer()

    start = time.time()

    X_train_vec = vectorizer.fit_transform(
        X_train
    )

    print(
        f"Training vectorization: "
        f"{time.time() - start:.2f} seconds"
    )

    start = time.time()

    X_test_vec = vectorizer.transform(
        X_test
    )

    print(
        f"Testing vectorization: "
        f"{time.time() - start:.2f} seconds"
    )

    print(
        f"\nFeature matrix: "
        f"{X_train_vec.shape}"
    )

    print(
        f"Total features: "
        f"{X_train_vec.shape[1]:,}"
    )

    # ========================================================
    # MODELS
    # ========================================================

    models = {

        "Logistic Regression":
            LogisticRegression(
                max_iter=1000,
                C=2.0,
                class_weight="balanced",
                n_jobs=-1
            ),

        "Linear SVM":
            LinearSVC(
                C=1.5,
                class_weight="balanced"
            ),

        "Complement Naive Bayes":
            ComplementNB(
                alpha=0.5
            )
    }

    # ========================================================
    # TRAIN MODELS
    # ========================================================

    results = []

    trained_models = {}

    for name, model in models.items():

        print("\n" + "-" * 70)
        print(
            f"Training: {name}"
        )
        print("-" * 70)

        start = time.time()

        model.fit(
            X_train_vec,
            y_train
        )

        elapsed = (
            time.time() - start
        )

        # ----------------------------------------------------
        # Evaluate
        # ----------------------------------------------------

        metrics = evaluate(
            model,
            X_test_vec,
            y_test
        )

        metrics["model"] = name

        metrics[
            "training_time_seconds"
        ] = elapsed

        results.append(
            metrics
        )

        trained_models[
            name
        ] = model

        print(
            f"Accuracy : "
            f"{metrics['accuracy']:.4f}"
        )

        print(
            f"Precision: "
            f"{metrics['precision']:.4f}"
        )

        print(
            f"Recall   : "
            f"{metrics['recall']:.4f}"
        )

        print(
            f"F1       : "
            f"{metrics['f1']:.4f}"
        )

        print(
            f"Training time: "
            f"{elapsed:.2f} seconds"
        )

    # ========================================================
    # MODEL COMPARISON
    # ========================================================

    results_df = pd.DataFrame(
        results
    )

    # Sort by F1 score
    results_df = results_df.sort_values(
        "f1",
        ascending=False
    ).reset_index(
        drop=True
    )

    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)

    print(
        results_df.to_string(
            index=False
        )
    )

    # ========================================================
    # CREATE OUTPUT DIRECTORIES
    # ========================================================

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # ========================================================
    # SAVE MODEL COMPARISON
    # ========================================================

    comparison_path = (
        REPORT_DIR /
        "model_comparison.csv"
    )

    results_df.to_csv(
        comparison_path,
        index=False
    )

    print(
        f"\nModel comparison saved:"
    )

    print(
        comparison_path
    )

    # ========================================================
    # SELECT BEST MODEL
    # ========================================================

    best_model_name = (
        results_df.iloc[0]["model"]
    )

    best_model = trained_models[
        best_model_name
    ]

    best_f1 = (
        results_df.iloc[0]["f1"]
    )

    print(
        "\n" + "=" * 70
    )

    print(
        f"Selected model: "
        f"{best_model_name}"
    )

    print(
        f"Best F1 score: "
        f"{best_f1:.4f}"
    )

    print(
        "=" * 70
    )

    # ========================================================
    # SAVE BEST MODEL
    # ========================================================

    joblib.dump(
        best_model,
        MODEL_FILE
    )

    print(
        "\nBest model saved:"
    )

    print(
        MODEL_FILE
    )

    # ========================================================
    # SAVE VECTORIZER
    # ========================================================

    joblib.dump(
        vectorizer,
        VECTORIZER_FILE
    )

    print(
        "\nTF-IDF vectorizer saved:"
    )

    print(
        VECTORIZER_FILE
    )

    # ========================================================
    # SAVE METADATA
    # ========================================================

    metadata = {

        "model": best_model_name,

        "features":
            "word_tfidf + character_tfidf",

        "word_features":
            250000,

        "character_features":
            150000,

        "random_state":
            RANDOM_STATE,

        "test_size":
            TEST_SIZE,

        "training_rows":
            len(X_train),

        "testing_rows":
            len(X_test),

        "total_features":
            int(X_train_vec.shape[1]),

        "metrics":
            results_df.iloc[0].to_dict()
    }

    with open(
        METADATA_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            metadata,
            f,
            indent=4,
            default=str
        )

    print(
        "\nMetadata saved:"
    )

    print(
        METADATA_FILE
    )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print("\n" + "=" * 70)
    print("TRAINING COMPLETED SUCCESSFULLY")
    print("=" * 70)

    print(
        "\nGenerated files:"
    )

    print(
        f"1. Model       : {MODEL_FILE}"
    )

    print(
        f"2. Vectorizer  : {VECTORIZER_FILE}"
    )

    print(
        f"3. Metadata    : {METADATA_FILE}"
    )

    print(
        f"4. Comparison  : {comparison_path}"
    )

    print("\nNext command:")

    print(
        "python src/05_evaluate_model.py"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()