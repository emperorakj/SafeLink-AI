import os
import re
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_PATH = os.path.join(
    BASE_DIR,
    "data",
    "scams.csv"
)

OUTPUT_PATH = os.path.join(
    BASE_DIR,
    "data",
    "scams_clean.csv"
)

CONFLICT_PATH = os.path.join(
    BASE_DIR,
    "data",
    "label_conflicts.csv"
)


TEXT_COLUMN = "text"
LABEL_COLUMN = "label"


def normalize_text(text):
    """
    Normalize text only for duplicate detection.
    The original text is preserved in the cleaned dataset.
    """

    text = str(text).lower()

    # Normalize placeholders
    text = re.sub(r"\[link\]", " LINK ", text)
    text = re.sub(r"\[phone\]", " PHONE ", text)
    text = re.sub(r"\[amount\]", " AMOUNT ", text)
    text = re.sub(r"\[bank\]", " BANK ", text)

    # Normalize URLs
    text = re.sub(
        r"https?://\S+|www\.\S+",
        " URL ",
        text
    )

    # Remove punctuation
    text = re.sub(
        r"[^\w\s₹]",
        " ",
        text
    )

    # Normalize whitespace
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


def main():

    print("=" * 65)
    print("              SAFELINK DATASET CLEANER")
    print("=" * 65)

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    if not os.path.exists(INPUT_PATH):

        print("\n❌ Dataset not found:")
        print(INPUT_PATH)
        return

    df = pd.read_csv(INPUT_PATH)

    print(f"\nOriginal rows: {len(df)}")

    # --------------------------------------------------------
    # BASIC VALIDATION
    # --------------------------------------------------------

    required_columns = {
        TEXT_COLUMN,
        LABEL_COLUMN
    }

    if not required_columns.issubset(df.columns):

        print("\n❌ Missing required columns.")

        print(
            "Required:",
            required_columns
        )

        print(
            "Found:",
            list(df.columns)
        )

        return

    # Remove completely empty rows
    df = df.dropna(
        subset=[
            TEXT_COLUMN,
            LABEL_COLUMN
        ]
    ).copy()

    # --------------------------------------------------------
    # NORMALIZATION
    # --------------------------------------------------------

    df["_normalized"] = df[
        TEXT_COLUMN
    ].apply(normalize_text)

    # --------------------------------------------------------
    # CHECK LABEL CONFLICTS
    # --------------------------------------------------------

    print("\nChecking label conflicts...")

    label_counts = (
        df.groupby("_normalized")[LABEL_COLUMN]
        .nunique()
    )

    conflicting_texts = label_counts[
        label_counts > 1
    ].index

    conflicts = df[
        df["_normalized"].isin(
            conflicting_texts
        )
    ].copy()

    if len(conflicts) > 0:

        print(
            f"⚠ Found {len(conflicts)} "
            "rows involved in label conflicts."
        )

        conflicts.sort_values(
            "_normalized"
        ).to_csv(
            CONFLICT_PATH,
            index=False
        )

        print(
            f"📄 Conflicts saved to:\n"
            f"{CONFLICT_PATH}"
        )

    else:

        print(
            "✅ No conflicting labels found."
        )

    # --------------------------------------------------------
    # REMOVE DUPLICATES
    # --------------------------------------------------------

    before = len(df)

    # Keep the first occurrence of each normalized message.
    # Conflicting-label examples are kept for manual review
    # rather than silently deleting them.
    clean_df = df[
        ~df["_normalized"].isin(
            conflicting_texts
        )
    ].copy()

    clean_df = clean_df.drop_duplicates(
        subset=["_normalized"],
        keep="first"
    )

    removed = before - len(clean_df)

    # --------------------------------------------------------
    # REMOVE INTERNAL COLUMN
    # --------------------------------------------------------

    clean_df = clean_df.drop(
        columns=["_normalized"]
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    clean_df.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8"
    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print("CLEANING COMPLETE")
    print("=" * 65)

    print(
        f"\nOriginal rows:       {before}"
    )

    print(
        f"Removed duplicates:  {removed}"
    )

    print(
        f"Conflict rows:       {len(conflicts)}"
    )

    print(
        f"Clean rows:          {len(clean_df)}"
    )

    print(
        f"\n📁 Clean dataset:"
        f"\n{OUTPUT_PATH}"
    )

    print("\nCLASS DISTRIBUTION")
    print("-" * 65)

    distribution = (
        clean_df[LABEL_COLUMN]
        .value_counts()
    )

    for label, count in distribution.items():

        percentage = (
            count /
            len(clean_df) *
            100
        )

        print(
            f"{label:<20}"
            f"{count:>5}"
            f" ({percentage:>6.2f}%)"
        )

    print("\n" + "=" * 65)


if __name__ == "__main__":
    main()