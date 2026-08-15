import os
import re
import pandas as pd
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "scams_clean.csv")

OUTPUT_REPORT = os.path.join(BASE_DIR, "dataset_audit_report.txt")


# ============================================================
# CONFIGURATION
# ============================================================

TEXT_COLUMN = "text"
LABEL_COLUMN = "label"

NEAR_DUPLICATE_THRESHOLD = 0.90


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_text(text):
    """
    Normalize text for duplicate detection.

    This does NOT modify the original dataset.
    It is only used to detect messages that are effectively
    the same despite capitalization, punctuation, spacing, etc.
    """

    text = str(text).lower()

    # Normalize common placeholders
    text = re.sub(r"\[link\]", " LINK ", text)
    text = re.sub(r"\[phone\]", " PHONE ", text)
    text = re.sub(r"\[amount\]", " AMOUNT ", text)
    text = re.sub(r"\[bank\]", " BANK ", text)

    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", " URL ", text)

    # Remove punctuation
    text = re.sub(r"[^\w\s₹]", " ", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Dataset not found:\n{DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    print(f"Dataset loaded: {DATA_PATH}")
    print(f"Rows found: {len(df)}")

    return df


# ============================================================
# BASIC VALIDATION
# ============================================================

def validate_dataset(df):
    issues = []

    if TEXT_COLUMN not in df.columns:
        issues.append(f"Missing column: {TEXT_COLUMN}")

    if LABEL_COLUMN not in df.columns:
        issues.append(f"Missing column: {LABEL_COLUMN}")

    if issues:
        print("\n❌ DATASET STRUCTURE ERROR")

        for issue in issues:
            print(f"   - {issue}")

        return False

    return True


# ============================================================
# EMPTY / MISSING VALUES
# ============================================================

def check_missing_values(df):
    empty_text = (
        df[TEXT_COLUMN]
        .isna()
        .sum()
    )

    empty_labels = (
        df[LABEL_COLUMN]
        .isna()
        .sum()
    )

    empty_text += (
        df[TEXT_COLUMN]
        .astype(str)
        .str.strip()
        .eq("")
        .sum()
    )

    return empty_text, empty_labels


# ============================================================
# EXACT DUPLICATES
# ============================================================

def find_exact_duplicates(df):
    duplicate_count = df.duplicated(
        subset=[TEXT_COLUMN],
        keep=False
    ).sum()

    duplicate_groups = (
        df[df.duplicated(
            subset=[TEXT_COLUMN],
            keep=False
        )]
        .groupby(TEXT_COLUMN)
        .size()
        .sort_values(ascending=False)
    )

    return duplicate_count, duplicate_groups


# ============================================================
# NORMALIZED DUPLICATES
# ============================================================

def find_normalized_duplicates(df):
    normalized = df[TEXT_COLUMN].apply(normalize_text)

    duplicate_count = normalized.duplicated(
        keep=False
    ).sum()

    duplicate_groups = (
        pd.DataFrame({
            "normalized": normalized
        })
        .groupby("normalized")
        .size()
        .sort_values(ascending=False)
    )

    return duplicate_count, duplicate_groups, normalized


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

def class_distribution(df):
    counts = Counter(
        df[LABEL_COLUMN]
        .astype(str)
        .str.strip()
    )

    return counts


# ============================================================
# LABEL VALIDATION
# ============================================================

def validate_labels(df):
    labels = (
        df[LABEL_COLUMN]
        .astype(str)
        .str.strip()
    )

    invalid = labels[
        labels.str.len() == 0
    ]

    return len(invalid)


# ============================================================
# NEAR DUPLICATE DETECTION
# ============================================================

def find_near_duplicates(df):
    """
    Detect highly similar messages using TF-IDF character
    n-grams and cosine similarity.

    This is intentionally separate from exact duplicates.

    Example:

        "Update KYC immediately"
        "Update your KYC immediately!"

    These are different strings but almost identical signals.
    """

    print("\n🔎 Checking near-duplicates...")

    texts = df[TEXT_COLUMN].astype(str).tolist()

    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        min_df=1,
        max_features=30000
    )

    matrix = vectorizer.fit_transform(texts)

    neighbors = NearestNeighbors(
        n_neighbors=2,
        metric="cosine",
        n_jobs=-1
    )

    neighbors.fit(matrix)

    distances, indices = neighbors.kneighbors(matrix)

    near_duplicate_pairs = []

    for i in range(len(texts)):
        nearest_index = indices[i][1]
        similarity = 1 - distances[i][1]

        if similarity >= NEAR_DUPLICATE_THRESHOLD:
            # Prevent storing the same pair twice
            pair = tuple(
                sorted((i, nearest_index))
            )

            near_duplicate_pairs.append(
                (
                    pair[0],
                    pair[1],
                    similarity
                )
            )

    near_duplicate_pairs = list(
        set(near_duplicate_pairs)
    )

    near_duplicate_pairs.sort(
        key=lambda x: x[2],
        reverse=True
    )

    return near_duplicate_pairs


# ============================================================
# REPORT GENERATION
# ============================================================

def generate_report(
    df,
    empty_text,
    empty_labels,
    exact_duplicate_count,
    exact_groups,
    normalized_duplicate_count,
    normalized_groups,
    distribution,
    invalid_labels,
    near_duplicate_pairs
):

    total_rows = len(df)

    report = []

    report.append("=" * 65)
    report.append("              SAFELINK DATASET AUDIT")
    report.append("=" * 65)

    report.append("")
    report.append("DATASET")
    report.append("-" * 65)
    report.append(f"Total rows:              {total_rows}")
    report.append(f"Columns:                 {', '.join(df.columns)}")

    report.append("")
    report.append("DATA QUALITY")
    report.append("-" * 65)
    report.append(f"Empty text rows:         {empty_text}")
    report.append(f"Empty label rows:        {empty_labels}")
    report.append(f"Invalid labels:          {invalid_labels}")
    report.append(f"Exact duplicate rows:    {exact_duplicate_count}")
    report.append(
        f"Normalized duplicate rows: {normalized_duplicate_count}"
    )
    report.append(
        f"Near-duplicate pairs:    {len(near_duplicate_pairs)}"
    )

    report.append("")
    report.append("CLASS DISTRIBUTION")
    report.append("-" * 65)

    for label, count in sorted(
        distribution.items(),
        key=lambda x: x[1],
        reverse=True
    ):
        percentage = (
            count / total_rows * 100
            if total_rows
            else 0
        )

        report.append(
            f"{label:<20} {count:>5} "
            f"({percentage:>6.2f}%)"
        )

    report.append("")
    report.append("MOST REPEATED NORMALIZED MESSAGES")
    report.append("-" * 65)

    for text, count in normalized_groups.head(15).items():

        if count <= 1:
            continue

        preview = text[:100]

        report.append(
            f"[{count}x] {preview}"
        )

    report.append("")
    report.append("TOP NEAR-DUPLICATE PAIRS")
    report.append("-" * 65)

    for i, j, similarity in near_duplicate_pairs[:20]:

        text_a = str(df.iloc[i][TEXT_COLUMN])
        text_b = str(df.iloc[j][TEXT_COLUMN])

        report.append(
            f"\nSimilarity: {similarity:.3f}"
        )

        report.append(
            f"A: {text_a[:150]}"
        )

        report.append(
            f"B: {text_b[:150]}"
        )

    # ========================================================
    # FINAL STATUS
    # ========================================================

    problems = []

    if empty_text > 0:
        problems.append("empty text")

    if empty_labels > 0:
        problems.append("empty labels")

    if invalid_labels > 0:
        problems.append("invalid labels")

    if normalized_duplicate_count > 0:
        problems.append("normalized duplicates")

    if near_duplicate_pairs:
        problems.append("near duplicates")

    report.append("")
    report.append("=" * 65)

    if problems:
        report.append("STATUS: ⚠ DATASET REQUIRES CLEANING")
        report.append("")
        report.append(
            "Issues detected: "
            + ", ".join(problems)
        )
    else:
        report.append(
            "STATUS: ✅ DATASET PASSED INITIAL AUDIT"
        )

    report.append("=" * 65)

    return "\n".join(report)


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n" + "=" * 65)
    print("              SAFELINK DATASET AUDIT")
    print("=" * 65)

    # Load
    df = load_dataset()

    # Validate structure
    if not validate_dataset(df):
        return

    # Missing values
    empty_text, empty_labels = check_missing_values(df)

    # Exact duplicates
    (
        exact_duplicate_count,
        exact_groups
    ) = find_exact_duplicates(df)

    # Normalized duplicates
    (
        normalized_duplicate_count,
        normalized_groups,
        normalized
    ) = find_normalized_duplicates(df)

    # Class distribution
    distribution = class_distribution(df)

    # Label validation
    invalid_labels = validate_labels(df)

    # Near duplicates
    near_duplicate_pairs = find_near_duplicates(df)

    # Generate report
    report = generate_report(
        df,
        empty_text,
        empty_labels,
        exact_duplicate_count,
        exact_groups,
        normalized_duplicate_count,
        normalized_groups,
        distribution,
        invalid_labels,
        near_duplicate_pairs
    )

    # Print report
    print("\n")
    print(report)

    # Save report
    with open(
        OUTPUT_REPORT,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(report)

    print(
        f"\n📄 Audit report saved to:\n"
        f"{OUTPUT_REPORT}"
    )


if __name__ == "__main__":
    main()