import os
import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TEST_PATH = os.path.join(
    BASE_DIR,
    "data",
    "splits",
    "test.csv"
)

MODEL_PATH = os.path.join(
    os.path.dirname(BASE_DIR),
    "backend",
    "models",
    "scam_model.pkl"
)

VECTORIZER_PATH = os.path.join(
    os.path.dirname(BASE_DIR),
    "backend",
    "models",
    "vectorizer.pkl"
)

REPORT_PATH = os.path.join(
    BASE_DIR,
    "evaluation_report.txt"
)


# ============================================================
# CONFIGURATION
# ============================================================

TEXT_COLUMN = "text"
LABEL_COLUMN = "label"


# ============================================================
# LOAD TEST DATA
# ============================================================

def load_test_data():

    if not os.path.exists(TEST_PATH):
        raise FileNotFoundError(
            f"Test dataset not found:\n{TEST_PATH}\n\n"
            "Run train_model.py first."
        )

    df = pd.read_csv(TEST_PATH)

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

    return df


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found:\n{MODEL_PATH}"
        )

    if not os.path.exists(VECTORIZER_PATH):
        raise FileNotFoundError(
            f"Vectorizer not found:\n{VECTORIZER_PATH}"
        )

    model = joblib.load(
        MODEL_PATH
    )

    vectorizer = joblib.load(
        VECTORIZER_PATH
    )

    return model, vectorizer


# ============================================================
# EVALUATE
# ============================================================

def evaluate(model, vectorizer, test_df):

    X_test = vectorizer.transform(
        test_df[TEXT_COLUMN]
    )

    y_test = test_df[
        LABEL_COLUMN
    ]

    predictions = model.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    labels = sorted(
        y_test.unique()
    )

    report = classification_report(
        y_test,
        predictions,
        labels=labels,
        zero_division=0
    )

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=labels
    )

    return (
        accuracy,
        report,
        matrix,
        labels,
        predictions
    )


# ============================================================
# PRINT CONFUSION MATRIX
# ============================================================

def print_confusion_matrix(
    matrix,
    labels
):

    print("\nCONFUSION MATRIX")
    print("-" * 100)

    # Header
    print(
        f"{'Actual / Predicted':<20}",
        end=""
    )

    for label in labels:
        print(
            f"{label[:12]:>14}",
            end=""
        )

    print()

    print("-" * 100)

    for i, label in enumerate(labels):

        print(
            f"{label:<20}",
            end=""
        )

        for value in matrix[i]:

            print(
                f"{value:>14}",
                end=""
            )

        print()


# ============================================================
# SAVE REPORT
# ============================================================

def save_report(
    accuracy,
    report,
    matrix,
    labels
):

    with open(
        REPORT_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "============================================================\n"
        )

        file.write(
            "              SAFELINK v0.3 EVALUATION\n"
        )

        file.write(
            "============================================================\n\n"
        )

        file.write(
            f"Test samples: {sum(matrix.flat)}\n"
        )

        file.write(
            f"Accuracy: {accuracy * 100:.2f}%\n\n"
        )

        file.write(
            "CLASSIFICATION REPORT\n"
        )

        file.write(
            "-" * 60 + "\n"
        )

        file.write(
            report
        )

        file.write(
            "\n\nCONFUSION MATRIX\n"
        )

        file.write(
            "-" * 60 + "\n"
        )

        file.write(
            f"{'Actual / Predicted':<20}"
        )

        for label in labels:

            file.write(
                f"{label[:12]:>14}"
            )

        file.write("\n")

        for i, label in enumerate(labels):

            file.write(
                f"{label:<20}"
            )

            for value in matrix[i]:

                file.write(
                    f"{value:>14}"
                )

            file.write("\n")

    print(
        f"\nEvaluation report saved to:\n"
        f"{REPORT_PATH}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n" + "=" * 65)
    print("              SAFELINK v0.3 EVALUATION")
    print("=" * 65)

    # Load test dataset
    test_df = load_test_data()

    print(
        f"\nTest samples: {len(test_df)}"
    )

    # Load model
    model, vectorizer = load_model()

    print(
        "\nModel and vectorizer loaded successfully."
    )

    # Evaluate
    (
        accuracy,
        report,
        matrix,
        labels,
        predictions
    ) = evaluate(
        model,
        vectorizer,
        test_df
    )

    # --------------------------------------------------------
    # Accuracy
    # --------------------------------------------------------

    print("\nTEST ACCURACY")
    print("-" * 65)

    print(
        f"{accuracy * 100:.2f}%"
    )

    # --------------------------------------------------------
    # Classification report
    # --------------------------------------------------------

    print("\nCLASSIFICATION REPORT")
    print("-" * 65)

    print(report)

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    print_confusion_matrix(
        matrix,
        labels
    )

    # --------------------------------------------------------
    # Save report
    # --------------------------------------------------------

    save_report(
        accuracy,
        report,
        matrix,
        labels
    )

    print("\n" + "=" * 65)
    print("✅ SAFELINK v0.3 EVALUATION COMPLETE")
    print("=" * 65)


if __name__ == "__main__":
    main()