from pathlib import Path

# ============================================================
# PROJECT PATHS
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

MODEL_DIR = ROOT_DIR / "models"

OUTPUT_DIR = ROOT_DIR / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
REPORT_DIR = OUTPUT_DIR / "reports"
PREDICTION_DIR = OUTPUT_DIR / "predictions"

# Create directories
for directory in [
    RAW_DIR,
    PROCESSED_DIR,
    MODEL_DIR,
    FIGURE_DIR,
    REPORT_DIR,
    PREDICTION_DIR
]:
    directory.mkdir(parents=True, exist_ok=True)


# ============================================================
# DATA
# ============================================================

RAW_FILE = RAW_DIR / "training.1600000.processed.noemoticon.csv"

PROCESSED_FILE = PROCESSED_DIR / "tweets_clean.csv"

PREDICTION_FILE = PREDICTION_DIR / "predictions.csv"


# ============================================================
# MODEL
# ============================================================

MODEL_FILE = MODEL_DIR / "sentiment_model.joblib"
VECTORIZER_FILE = MODEL_DIR / "tfidf_vectorizer.joblib"
METADATA_FILE = MODEL_DIR / "model_metadata.json"


# ============================================================
# LABELS
# ============================================================

LABEL_MAP = {
    0: "Negative",
    4: "Positive"
}

LABEL_TO_INT = {
    "Negative": 0,
    "Positive": 1
}


# ============================================================
# REPRODUCIBILITY
# ============================================================

RANDOM_STATE = 42


# ============================================================
# TRAINING
# ============================================================

TEST_SIZE = 0.20

# Set to None for the complete dataset.
# During experimentation you can use e.g. 300000.
TRAINING_SAMPLE_SIZE = None


# ============================================================
# VISUAL DESIGN
# ============================================================

COLORS = {
    "negative": "#E45756",
    "positive": "#2A9D8F",
    "neutral": "#F4A261",
    "primary": "#264653",
    "secondary": "#457B9D",
    "light": "#F1FAEE",
    "dark": "#1D3557"
}

SENTIMENT_COLORS = [
    COLORS["negative"],
    COLORS["positive"]
]