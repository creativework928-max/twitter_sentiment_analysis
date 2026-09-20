import pandas as pd
import numpy as np

from config import RAW_FILE


COLUMN_NAMES = [
    "target",
    "tweet_id",
    "date",
    "query",
    "user",
    "text"
]


def load_dataset():

    print("Loading dataset...")

    df = pd.read_csv(
        RAW_FILE,
        encoding="latin-1",
        header=None,
        names=COLUMN_NAMES
    )

    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    return df


def validate_dataset(df):

    print("\n" + "=" * 60)
    print("DATASET VALIDATION")
    print("=" * 60)

    print("\nData types:")
    print(df.dtypes)

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nDuplicate rows:")
    print(df.duplicated().sum())

    print("\nTarget distribution:")
    print(df["target"].value_counts())

    print("\nDataset shape:")
    print(df.shape)

    print("\nSample:")
    print(df.head())


def main():

    df = load_dataset()

    validate_dataset(df)


if __name__ == "__main__":
    main()