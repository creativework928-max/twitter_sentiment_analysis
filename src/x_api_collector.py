import os
import requests
import pandas as pd

from dotenv import load_dotenv


load_dotenv()

BEARER_TOKEN = os.getenv(
    "X_BEARER_TOKEN"
)


URL = (
    "https://api.x.com/2/tweets/search/recent"
)


def search_posts(
    query,
    max_results=100
):

    if not BEARER_TOKEN:

        raise ValueError(
            "X_BEARER_TOKEN is not configured."
        )

    headers = {
        "Authorization":
            f"Bearer {BEARER_TOKEN}"
    }

    params = {

        "query": query,

        "max_results": max_results,

        "tweet.fields":
            "id,text,created_at,public_metrics"
    }

    response = requests.get(
        URL,
        headers=headers,
        params=params,
        timeout=30
    )

    response.raise_for_status()

    data = response.json()

    posts = data.get(
        "data",
        []
    )

    return pd.DataFrame(
        posts
    )


if __name__ == "__main__":

    query = (
        '(AI OR "artificial intelligence") '
        'lang:en -is:retweet'
    )

    df = search_posts(
        query
    )

    print(df.head())

    df.to_csv(
        "current_x_posts.csv",
        index=False
    )