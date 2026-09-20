import shutil
from pathlib import Path

import kagglehub

from config import RAW_DIR, RAW_FILE


DATASET = "kazanova/sentiment140"


def find_csv(folder):
    folder = Path(folder)

    csv_files = list(folder.rglob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(
            "No CSV file found in downloaded dataset."
        )

    # Prefer original Sentiment140 training file
    for file in csv_files:
        if "training.1600000" in file.name.lower():
            return file

    return csv_files[0]


def main():

    print("=" * 70)
    print("Downloading Sentiment140")
    print("=" * 70)

    download_path = kagglehub.dataset_download(DATASET)

    print(f"Dataset downloaded to: {download_path}")

    source_file = find_csv(download_path)

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    shutil.copy2(source_file, RAW_FILE)

    print(f"\nDataset copied to:")
    print(RAW_FILE)

    print("\nDownload completed successfully.")


if __name__ == "__main__":
    main()