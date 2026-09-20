import sys
from pathlib import Path

import joblib

from fastapi import FastAPI
from pydantic import BaseModel


ROOT_DIR = Path(__file__).resolve().parents[1]

MODEL_FILE = (
    ROOT_DIR /
    "models" /
    "sentiment_model.joblib"
)

VECTORIZER_FILE = (
    ROOT_DIR /
    "models" /
    "tfidf_vectorizer.joblib"
)


model = joblib.load(
    MODEL_FILE
)

vectorizer = joblib.load(
    VECTORIZER_FILE
)


app = FastAPI(
    title="Twitter Sentiment Analysis API",
    description=(
        "Production sentiment classification "
        "API for Twitter/X text."
    ),
    version="1.0.0"
)


class TweetRequest(BaseModel):

    text: str


class TweetResponse(BaseModel):

    text: str
    sentiment: str
    sentiment_score: float | None = None


@app.get("/")
def root():

    return {
        "project": "Twitter Sentiment Analysis",
        "status": "running"
    }


@app.post(
    "/predict",
    response_model=TweetResponse
)
def predict(request: TweetRequest):

    text = request.text

    vector = vectorizer.transform(
        [text]
    )

    prediction = model.predict(
        vector
    )[0]

    sentiment = (
        "Positive"
        if prediction == 1
        else "Negative"
    )

    score = None

    if hasattr(model, "decision_function"):

        score = float(
            model.decision_function(vector)[0]
        )

    return TweetResponse(
        text=text,
        sentiment=sentiment,
        sentiment_score=score
    )