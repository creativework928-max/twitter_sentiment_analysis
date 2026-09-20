from pathlib import Path

import streamlit as st
import pandas as pd
import joblib

import plotly.express as px


ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    ROOT /
    "models" /
    "sentiment_model.joblib"
)

VECTORIZER_PATH = (
    ROOT /
    "models" /
    "tfidf_vectorizer.joblib"
)

DATA_PATH = (
    ROOT /
    "data" /
    "processed" /
    "tweets_clean.csv"
)

FIGURE_DIR = (
    ROOT /
    "outputs" /
    "figures"
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Twitter Sentiment Intelligence",
    page_icon="📊",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main {
        background-color: #F8FAFC;
    }

    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
    }

    .title {
        color: #264653;
        font-size: 42px;
        font-weight: 800;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="title">Twitter Sentiment Intelligence</div>',
    unsafe_allow_html=True
)

st.write(
    """
    End-to-end sentiment analysis platform for Twitter/X-style
    social media text using machine learning and NLP.
    """
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    model = joblib.load(
        MODEL_PATH
    )

    vectorizer = joblib.load(
        VECTORIZER_PATH
    )

    return model, vectorizer


model, vectorizer = load_model()


# ============================================================
# PREDICTION
# ============================================================

st.header("Live Tweet Sentiment")

tweet = st.text_area(
    "Enter a tweet",
    placeholder=(
        "Example: I really love this new product!"
    ),
    height=130
)


if st.button(
    "Analyze Sentiment",
    type="primary"
):

    if not tweet.strip():

        st.warning(
            "Please enter a tweet."
        )

    else:

        vector = vectorizer.transform(
            [tweet]
        )

        prediction = model.predict(
            vector
        )[0]

        if prediction == 1:

            sentiment = "Positive"
            color = "#2A9D8F"

        else:

            sentiment = "Negative"
            color = "#E45756"

        st.markdown(
            f"""
            <div style="
                background:{color};
                color:white;
                padding:25px;
                border-radius:15px;
                text-align:center;
                font-size:30px;
                font-weight:bold;
            ">
                {sentiment}
            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# DATASET ANALYTICS
# ============================================================

st.header("Dataset Analytics")

df = pd.read_csv(
    DATA_PATH
)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Tweets",
        f"{len(df):,}"
    )

with col2:
    st.metric(
        "Positive",
        f"{(df['sentiment'] == 'Positive').sum():,}"
    )

with col3:
    st.metric(
        "Negative",
        f"{(df['sentiment'] == 'Negative').sum():,}"
    )

with col4:
    st.metric(
        "Average Words",
        f"{df['word_count'].mean():.1f}"
    )


# ============================================================
# SENTIMENT CHART
# ============================================================

counts = (
    df["sentiment"]
    .value_counts()
    .reset_index()
)

counts.columns = [
    "sentiment",
    "count"
]

fig = px.pie(
    counts,
    names="sentiment",
    values="count",
    title="Overall Sentiment Distribution",
    color="sentiment",
    color_discrete_map={
        "Positive": "#2A9D8F",
        "Negative": "#E45756"
    },
    hole=0.45
)

st.plotly_chart(
    fig,
    use_container_width=True
)


# ============================================================
# STATIC PROFESSIONAL FIGURES
# ============================================================

st.header("Project Visualizations")

figure_files = [
    "01_class_distribution.png",
    "02_tweet_length_distribution.png",
    "04_positive_wordcloud.png",
    "05_negative_wordcloud.png",
    "06_sentiment_over_time.png",
    "07_confusion_matrix.png",
    "08_roc_curve.png",
    "09_topic_sentiment.png",
    "10_topic_sentiment_heatmap.png"
]

for figure in figure_files:

    path = FIGURE_DIR / figure

    if path.exists():

        st.image(
            str(path),
            caption=figure.replace(
                "_",
                " "
            ).replace(
                ".png",
                ""
            ).title(),
            use_container_width=True
        )