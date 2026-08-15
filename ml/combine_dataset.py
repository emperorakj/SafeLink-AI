import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

CLEAN_PATH = os.path.join(DATA_DIR, "scams_clean.csv")
HARD_PATH = os.path.join(DATA_DIR, "hard_examples.csv")
OUTPUT_PATH = os.path.join(DATA_DIR, "scams_v031.csv")

TEXT_COLUMN = "text"
LABEL_COLUMN = "label"


def main():
    print("=" * 65)
    print("           SAFELINK v0.3.1 DATASET BUILDER")
    print("=" * 65)

    if not os.path.exists(CLEAN_PATH):
        raise FileNotFoundError(f"Missing: {CLEAN_PATH}")

    if not os.path.exists(HARD_PATH):
        raise FileNotFoundError(f"Missing: {HARD_PATH}")

    clean_df = pd.read_csv(CLEAN_PATH)
    hard_df = pd.read_csv(HARD_PATH)

    print(f"\nClean dataset:  {len(clean_df)}")
    print(f"Hard examples:  {len(hard_df)}")

    df = pd.concat(
        [clean_df, hard_df],
        ignore_index=True
    )

    # Basic cleanup
    df = df.dropna(
        subset=[TEXT_COLUMN, LABEL_COLUMN]
    ).copy()

    df[TEXT_COLUMN] = (
        df[TEXT_COLUMN]
        .astype(str)
        .str.strip()
    )

    df[LABEL_COLUMN] = (
        df[LABEL_COLUMN]
        .astype(str)
        .str.strip()
    )

    df = df[df[TEXT_COLUMN] != ""]

    # Exact duplicate removal only here.
    # scams_clean.csv was already normalized-deduplicated.
    before = len(df)

    df = df.drop_duplicates(
        subset=[TEXT_COLUMN],
        keep="first"
    ).reset_index(drop=True)

    removed = before - len(df)

    df.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8"
    )

    print("\n" + "-" * 65)
    print("v0.3.1 DATASET")
    print("-" * 65)

    print(f"Combined rows:       {before}")
    print(f"Exact duplicates:    {removed}")
    print(f"Final rows:          {len(df)}")

    print("\nCLASS DISTRIBUTION")
    print("-" * 65)

    counts = df[LABEL_COLUMN].value_counts()

    for label, count in counts.items():
        pct = count / len(df) * 100
        print(f"{label:<20}{count:>5} ({pct:>6.2f}%)")

    print("\n" + "=" * 65)
    print("✅ v0.3.1 DATASET CREATED")
    print("=" * 65)
    print(f"\nSaved to:\n{OUTPUT_PATH}")


if __name__ == "__main__":
    main()
