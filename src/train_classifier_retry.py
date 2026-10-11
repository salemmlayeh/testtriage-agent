"""Step 3b: does the retry result help the classifier?

Compares two models on the SAME train/test split:
  A) error message only
  B) error message + retry result

Run from the project root:  python src/train_classifier_retry.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# Reuse the functions from the first script (same folder)
from train_classifier import ROOT, SEED, build_pipeline, load_failures, normalize

IMAGE_FILE = ROOT / "docs" / "images" / "03b_confusion_matrix_with_retry.png"


def build_pipeline_with_retry():
    """Two kinds of input: the error text and the retry result (pass/fail)."""
    features = ColumnTransformer([
        ("text", TfidfVectorizer(preprocessor=normalize, ngram_range=(1, 2)), "error_message"),
        ("retry", OneHotEncoder(handle_unknown="ignore"), ["retry_result"]),
    ])
    return Pipeline([
        ("features", features),
        ("clf", LogisticRegression(max_iter=1000)),
    ])


def main():
    failures = load_failures()
    y = failures["failure_type"]
    X = failures[["error_message", "retry_result"]]  # a table with 2 columns

    # Same settings as before, so the test set contains the same failures
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=SEED, stratify=y
    )

    # Model A: text only (the model from the first script)
    model_a = build_pipeline()
    model_a.fit(X_train["error_message"], y_train)
    pred_a = model_a.predict(X_test["error_message"])

    # Model B: text + retry result
    model_b = build_pipeline_with_retry()
    model_b.fit(X_train, y_train)
    pred_b = model_b.predict(X_test)

    print("=== A) Text only ===")
    print(classification_report(y_test, pred_a, digits=3))
    print("=== B) Text + retry result ===")
    print(classification_report(y_test, pred_b, digits=3))

    # Cross-validated macro F1 for both (more stable than one split)
    cv_a = cross_val_score(build_pipeline(), X["error_message"], y, cv=5, scoring="f1_macro")
    cv_b = cross_val_score(build_pipeline_with_retry(), X, y, cv=5, scoring="f1_macro")

    rep_a = classification_report(y_test, pred_a, output_dict=True)
    rep_b = classification_report(y_test, pred_b, output_dict=True)

    print("=== Summary (before -> after) ===")
    print(f"Accuracy:               {rep_a['accuracy']:.3f} -> {rep_b['accuracy']:.3f}")
    print(f"Flaky recall:           {rep_a['flaky']['recall']:.3f} -> {rep_b['flaky']['recall']:.3f}")
    print(f"Communication precision:{rep_a['communication']['precision']:.3f} -> {rep_b['communication']['precision']:.3f}")
    print(f"Cross-validated F1:     {cv_a.mean():.3f} -> {cv_b.mean():.3f}")

    # Save the confusion matrix of model B for the README
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay.from_predictions(
        y_test, pred_b, ax=ax, cmap="Blues", xticks_rotation=30
    )
    ax.set_title("Classifier with retry result: confusion matrix")
    fig.tight_layout()
    fig.savefig(IMAGE_FILE, dpi=150)
    print(f"\nSaved confusion matrix to {IMAGE_FILE}")


if __name__ == "__main__":
    main()