import os
import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import NearestNeighbors


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "scams_v031.csv"
)

SPLIT_DIR = os.path.join(
    BASE_DIR,
    "data",
    "splits"
)

MODEL_DIR = os.path.join(
    os.path.dirname(BASE_DIR),
    "backend",
    "models"
)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "scam_model.pkl"
)

VECTORIZER_PATH = os.path.join(
    MODEL_DIR,
    "vectorizer.pkl"
)


# ============================================================
# CONFIGURATION
# ============================================================

TEXT_COLUMN = "text"
LABEL_COLUMN = "label"

# Messages above this similarity are considered part
# of the same linguistic group.
GROUP_SIMILARITY_THRESHOLD = 0.97

RANDOM_STATE = 42


# ============================================================
# CREATE DIRECTORIES
# ============================================================

os.makedirs(SPLIT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset():

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Clean dataset not found:\n{DATA_PATH}\n\n"
            "Run clean_dataset.py first."
        )

    df = pd.read_csv(DATA_PATH)

    if TEXT_COLUMN not in df.columns:
        raise ValueError(
            f"Missing column: {TEXT_COLUMN}"
        )

    if LABEL_COLUMN not in df.columns:
        raise ValueError(
            f"Missing column: {LABEL_COLUMN}"
        )

    df = df.dropna(
        subset=[
            TEXT_COLUMN,
            LABEL_COLUMN
        ]
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

    df = df[
        df[TEXT_COLUMN] != ""
    ].reset_index(drop=True)

    return df


# ============================================================
# CREATE SIMILARITY GROUPS
# ============================================================

def create_similarity_groups(df):

    print("\nCreating similarity groups...")

    texts = df[TEXT_COLUMN].tolist()

    # Character n-grams are useful for:
    # - spelling variations
    # - Hinglish
    # - abbreviations
    # - punctuation differences
    # - small message modifications

    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        min_df=1,
        max_features=30000
    )

    matrix = vectorizer.fit_transform(texts)

    neighbors = NearestNeighbors(
        radius=1 - GROUP_SIMILARITY_THRESHOLD,
        metric="cosine",
        n_jobs=-1
    )

    neighbors.fit(matrix)

    distances, indices = neighbors.radius_neighbors(
        matrix
    )

    # --------------------------------------------------------
    # Union-Find
    # --------------------------------------------------------

    parent = list(range(len(texts)))

    def find(x):

        while parent[x] != x:

            parent[x] = parent[parent[x]]
            x = parent[x]

        return x

    def union(a, b):

        root_a = find(a)
        root_b = find(b)

        if root_a != root_b:
            parent[root_b] = root_a

    # Connect highly similar messages
    for i in range(len(texts)):

        for distance, j in zip(
            distances[i],
            indices[i]
        ):

            if i == j:
                continue

            similarity = 1 - distance

            if similarity >= GROUP_SIMILARITY_THRESHOLD:
                union(i, j)

    # Convert roots to compact group IDs
    roots = {}

    groups = []

    for i in range(len(texts)):

        root = find(i)

        if root not in roots:
            roots[root] = len(roots)

        groups.append(
            roots[root]
        )

    print(
        f"Total examples: {len(texts)}"
    )

    print(
        f"Similarity groups: {len(set(groups))}"
    )

    largest_group = max(
        pd.Series(groups).value_counts()
    )

    print(
        f"Largest similarity group: {largest_group}"
    )

    return groups


# ============================================================
# STRATIFIED GROUP SPLIT
# ============================================================

def create_splits(df, groups):

    print("\nCreating train / validation / test split...")

    X = df[TEXT_COLUMN]
    y = df[LABEL_COLUMN]

    groups = pd.Series(
        groups,
        index=df.index
    )

    # --------------------------------------------------------
    # First split:
    #
    # 80% temporary training data
    # 20% test data
    # --------------------------------------------------------

    splitter = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    train_val_idx = None
    test_idx = None

    for train_idx, candidate_test_idx in splitter.split(
        X,
        y,
        groups
    ):

        train_val_idx = train_idx
        test_idx = candidate_test_idx
        break

    # --------------------------------------------------------
    # Second split:
    #
    # Remaining 80%
    #   ├── 80% train
    #   └── 20% validation
    #
    # Final approximately:
    #
    # Train:      64%
    # Validation: 16%
    # Test:       20%
    # --------------------------------------------------------

    X_train_val = X.iloc[
        train_val_idx
    ]

    y_train_val = y.iloc[
        train_val_idx
    ]

    groups_train_val = groups.iloc[
        train_val_idx
    ]

    second_splitter = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    train_idx_relative = None
    val_idx_relative = None

    for candidate_train_idx, candidate_val_idx in second_splitter.split(
        X_train_val,
        y_train_val,
        groups_train_val
    ):

        train_idx_relative = candidate_train_idx
        val_idx_relative = candidate_val_idx
        break

    # Convert relative indexes back to original indexes
    train_idx = train_val_idx[
        train_idx_relative
    ]

    val_idx = train_val_idx[
        val_idx_relative
    ]

    train_df = df.iloc[
        train_idx
    ].copy()

    val_df = df.iloc[
        val_idx
    ].copy()

    test_df = df.iloc[
        test_idx
    ].copy()

    # --------------------------------------------------------
    # Save splits
    # --------------------------------------------------------

    train_path = os.path.join(
        SPLIT_DIR,
        "train.csv"
    )

    val_path = os.path.join(
        SPLIT_DIR,
        "validation.csv"
    )

    test_path = os.path.join(
        SPLIT_DIR,
        "test.csv"
    )

    train_df.to_csv(
        train_path,
        index=False
    )

    val_df.to_csv(
        val_path,
        index=False
    )

    test_df.to_csv(
        test_path,
        index=False
    )

    print("\nSPLIT RESULTS")
    print("-" * 65)

    total = len(df)

    print(
        f"Training:      {len(train_df):>5}"
        f" ({len(train_df) / total * 100:.2f}%)"
    )

    print(
        f"Validation:    {len(val_df):>5}"
        f" ({len(val_df) / total * 100:.2f}%)"
    )

    print(
        f"Test:          {len(test_df):>5}"
        f" ({len(test_df) / total * 100:.2f}%)"
    )

    print("\nSplit files saved:")
    print(train_path)
    print(val_path)
    print(test_path)

    return train_df, val_df, test_df


# ============================================================
# TRAIN MODEL
# ============================================================

def train_model(train_df):

    print("\nTraining TF-IDF + Logistic Regression model...")

    # IMPORTANT:
    # TF-IDF is fitted ONLY on training data.
    #
    # Validation and test data remain unseen during training.

    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        min_df=2,
        max_features=30000,
        sublinear_tf=True
    )

    X_train = vectorizer.fit_transform(
        train_df[TEXT_COLUMN]
    )

    y_train = train_df[
        LABEL_COLUMN
    ]

    print(
        f"Training matrix: "
        f"{X_train.shape[0]} samples × "
        f"{X_train.shape[1]} features"
    )

    model = LogisticRegression(
        max_iter=3000,
        class_weight="balanced",
        random_state=RANDOM_STATE
    )

    model.fit(
        X_train,
        y_train
    )

    return model, vectorizer


# ============================================================
# VALIDATION
# ============================================================

def validate_model(
    model,
    vectorizer,
    val_df
):

    print("\nValidation performance...")

    X_val = vectorizer.transform(
        val_df[TEXT_COLUMN]
    )

    y_val = val_df[
        LABEL_COLUMN
    ]

    accuracy = model.score(
        X_val,
        y_val
    )

    print(
        f"Validation accuracy: "
        f"{accuracy * 100:.2f}%"
    )


# ============================================================
# SAVE MODEL
# ============================================================

def save_model(
    model,
    vectorizer
):

    joblib.dump(
        model,
        MODEL_PATH
    )

    joblib.dump(
        vectorizer,
        VECTORIZER_PATH
    )

    print("\nMODEL SAVED")
    print("-" * 65)

    print(
        f"Model:      {MODEL_PATH}"
    )

    print(
        f"Vectorizer: {VECTORIZER_PATH}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n" + "=" * 65)
    print("              SAFELINK v0.3 MODEL TRAINING")
    print("=" * 65)

    # Load
    df = load_dataset()

    print(
        f"\nLoaded clean dataset: "
        f"{len(df)} examples"
    )

    # Show classes
    print("\nCLASS DISTRIBUTION")
    print("-" * 65)

    distribution = (
        df[LABEL_COLUMN]
        .value_counts()
    )

    for label, count in distribution.items():

        percentage = (
            count /
            len(df) *
            100
        )

        print(
            f"{label:<20}"
            f"{count:>5}"
            f" ({percentage:>6.2f}%)"
        )

    # Similarity groups
    groups = create_similarity_groups(
        df
    )

    # Train / validation / test
    train_df, val_df, test_df = create_splits(
        df,
        groups
    )

    # Train
    model, vectorizer = train_model(
        train_df
    )

    # Validation
    validate_model(
        model,
        vectorizer,
        val_df
    )

    # Save
    save_model(
        model,
        vectorizer
    )

    print("\n" + "=" * 65)
    print("✅ SAFELINK v0.3 TRAINING COMPLETE")
    print("=" * 65)

    print(
        "\nNext step:"
        "\nRun evaluate.py against the saved test set."
    )


if __name__ == "__main__":
    main()