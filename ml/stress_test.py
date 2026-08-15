import os
import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CHALLENGE_PATH = os.path.join(
    BASE_DIR, "data", "challenge_set.csv"
)

MODEL_PATH = os.path.join(
    os.path.dirname(BASE_DIR),
    "backend", "models", "scam_model.pkl"
)

VECTORIZER_PATH = os.path.join(
    os.path.dirname(BASE_DIR),
    "backend", "models", "vectorizer.pkl"
)

REPORT_PATH = os.path.join(
    BASE_DIR, "challenge_report.txt"
)

TEXT_COLUMN = "text"
LABEL_COLUMN = "label"


def main():
    print("\n" + "=" * 70)
    print("              SAFELINK v0.3 ROBUSTNESS TEST")
    print("=" * 70)

    df = pd.read_csv(CHALLENGE_PATH)

    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)

    X = vectorizer.transform(df[TEXT_COLUMN])
    y_true = df[LABEL_COLUMN]

    y_pred = model.predict(X)

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)
        confidence = probabilities.max(axis=1)
    else:
        confidence = [None] * len(df)

    accuracy = accuracy_score(y_true, y_pred)

    print(f"\nChallenge samples: {len(df)}")
    print(f"Challenge accuracy: {accuracy * 100:.2f}%")

    print("\nCLASSIFICATION REPORT")
    print("-" * 70)
    print(classification_report(
        y_true,
        y_pred,
        zero_division=0
    ))

    labels = sorted(df[LABEL_COLUMN].unique())
    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=labels
    )

    print("CONFUSION MATRIX")
    print("-" * 100)
    print(f"{'Actual / Predicted':<20}", end="")
    for label in labels:
        print(f"{label[:12]:>14}", end="")
    print()

    for i, label in enumerate(labels):
        print(f"{label:<20}", end="")
        for value in matrix[i]:
            print(f"{value:>14}", end="")
        print()

    # Show every failure
    failures = df[y_true != y_pred].copy()
    failure_indices = failures.index

    print("\nMISCLASSIFICATIONS")
    print("-" * 70)

    if failures.empty:
        print("None. The model classified every challenge example correctly.")
    else:
        for idx in failure_indices:
            print(f"\nText:      {df.loc[idx, TEXT_COLUMN]}")
            print(f"Expected:  {df.loc[idx, LABEL_COLUMN]}")
            print(f"Predicted: {y_pred[idx]}")
            if confidence[idx] is not None:
                print(f"Confidence: {confidence[idx] * 100:.2f}%")

    # Confidence summary
    print("\nCONFIDENCE SUMMARY")
    print("-" * 70)
    print(f"Mean confidence: {sum(confidence) / len(confidence) * 100:.2f}%")
    print(f"Minimum confidence: {min(confidence) * 100:.2f}%")
    print(f"Maximum confidence: {max(confidence) * 100:.2f}%")

    # Save a machine-readable result file
    results = df.copy()
    results["predicted_label"] = y_pred
    results["correct"] = (
        results[LABEL_COLUMN] == results["predicted_label"]
    )
    results["confidence"] = confidence

    results.to_csv(
        os.path.join(BASE_DIR, "challenge_predictions.csv"),
        index=False
    )

    # Save text report
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("SAFELINK v0.3 ROBUSTNESS TEST\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Challenge samples: {len(df)}\n")
        f.write(f"Accuracy: {accuracy * 100:.2f}%\n\n")
        f.write(classification_report(
            y_true,
            y_pred,
            zero_division=0
        ))
        f.write("\n\nMISCLASSIFICATIONS\n")
        f.write("-" * 70 + "\n")

        if failures.empty:
            f.write("None\n")
        else:
            for idx in failure_indices:
                f.write(
                    f"\nText: {df.loc[idx, TEXT_COLUMN]}\n"
                    f"Expected: {df.loc[idx, LABEL_COLUMN]}\n"
                    f"Predicted: {y_pred[idx]}\n"
                    f"Confidence: {confidence[idx] * 100:.2f}%\n"
                )

    print("\nResults saved:")
    print(os.path.join(BASE_DIR, "challenge_predictions.csv"))
    print(REPORT_PATH)

    print("\n" + "=" * 70)
    print("✅ SAFELINK v0.3 ROBUSTNESS TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()