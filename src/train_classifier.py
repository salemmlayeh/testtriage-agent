"""Step 3: predict the failure type from the error message.

Run from the project root:  python src/train_classifier.py
"""
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # draw to a file, no window needed
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "data" / "test_results.csv"
IMAGE_FILE = ROOT / "docs" / "images" / "03_confusion_matrix.png"
SEED = 42


def normalize(text):
    """Lowercase and replace changing values with placeholders."""
    text = text.lower()
    text = re.sub(r"0x[0-9a-f]+", " hexid ", text)   # CAN ids like 0x7c7
    text = re.sub(r"\d+(?:\.\d+)?", " num ", text)   # any other number
    return text


def load_failures():
    """Load the CSV and keep only the failed tests."""
    df = pd.read_csv(DATA_FILE)
    return df[df["result"] == "fail"].reset_index(drop=True)


def build_pipeline():
    """Text -> TF-IDF numbers -> Logistic Regression."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(preprocessor=normalize, ngram_range=(1, 2))),
        ("clf", LogisticRegression(max_iter=1000)),
    ])


def main():
    failures = load_failures()
    X = failures["error_message"]
    y = failures["failure_type"]
    print(f"Failed tests: {len(failures)}\n")

    # Keep 25% of the data hidden for an honest evaluation.
    # stratify=y keeps the same mix of failure types in both parts.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=SEED, stratify=y
    )

    model = build_pipeline()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(f"Accuracy on hidden test set: {accuracy_score(y_test, y_pred):.3f}\n")
    print(classification_report(y_test, y_pred, digits=3))

    # 5-fold cross-validation: a more stable estimate on such a small dataset
    scores = cross_val_score(build_pipeline(), X, y, cv=5, scoring="f1_macro")
    print(f"Cross-validated macro F1: {scores.mean():.3f} (+/- {scores.std():.3f})\n")

    # Save the confusion matrix as a picture for the README
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, ax=ax, cmap="Blues", xticks_rotation=30
    )
    ax.set_title("Failure type classifier: confusion matrix")
    fig.tight_layout()
    IMAGE_FILE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(IMAGE_FILE, dpi=150)
    print(f"Saved confusion matrix to {IMAGE_FILE}")


if __name__ == "__main__":
    main()