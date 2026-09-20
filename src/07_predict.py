import sys

import joblib

from config import (
    MODEL_FILE,
    VECTORIZER_FILE
)


class TwitterSentimentPredictor:

    def __init__(self):

        self.model = joblib.load(
            MODEL_FILE
        )

        self.vectorizer = joblib.load(
            VECTORIZER_FILE
        )

    def predict(self, tweets):

        if isinstance(tweets, str):
            tweets = [tweets]

        vectors = self.vectorizer.transform(
            tweets
        )

        predictions = self.model.predict(
            vectors
        )

        results = []

        for tweet, prediction in zip(
            tweets,
            predictions
        ):

            sentiment = (
                "Positive"
                if prediction == 1
                else "Negative"
            )

            results.append({
                "tweet": tweet,
                "sentiment": sentiment
            })

        return results


def main():

    predictor = TwitterSentimentPredictor()

    if len(sys.argv) > 1:

        tweet = " ".join(
            sys.argv[1:]
        )

    else:

        tweet = input(
            "Enter a tweet: "
        )

    result = predictor.predict(
        tweet
    )

    print("\nPrediction:")
    print(result[0])


if __name__ == "__main__":
    main()