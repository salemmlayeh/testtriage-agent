"""Step 4: group failures WITHOUT using the labels (clustering).

Run from the project root:  python src/cluster_failures.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    adjusted_rand_score,
    completeness_score,
    homogeneity_score,
    silhouette_score,
)

# Reuse the helpers from the classifier script
from train_classifier import ROOT, SEED, load_failures, normalize

IMAGE_FILE = ROOT / "docs" / "images" / "04_clusters.png"
N_CLUSTERS = 4


def vectorize(messages):
    """Error messages -> TF-IDF numbers."""
    vectorizer = TfidfVectorizer(preprocessor=normalize, ngram_range=(1, 2))
    X = vectorizer.fit_transform(messages)
    return vectorizer, X


def cluster_failures(messages, k=N_CLUSTERS):
    """Group messages into k clusters. Step 5 will reuse this as an agent tool."""
    vectorizer, X = vectorize(messages)
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=SEED)
    labels = kmeans.fit_predict(X)
    return vectorizer, X, kmeans, labels


def print_top_terms(vectorizer, kmeans, n=5):
    """Show the most typical words of each cluster."""
    terms = np.array(vectorizer.get_feature_names_out())
    for i, center in enumerate(kmeans.cluster_centers_):
        top = terms[center.argsort()[::-1][:n]]
        print(f"  Cluster {i}: {', '.join(top)}")


def main():
    failures = load_failures()
    messages = failures["error_message"]
    true_types = failures["failure_type"]  # only used for checking, never for clustering
    print(f"Failed tests: {len(failures)}\n")

    # 1) Which number of clusters looks best? (silhouette needs no labels)
    _, X = vectorize(messages)
    print("Silhouette score for different numbers of clusters:")
    for k in range(2, 11):
        labels_k = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit_predict(X)
        print(f"  k={k:2d}: {silhouette_score(X, labels_k):.3f}")

    # 2) Cluster with k=4 (we know there are 4 failure types)
    vectorizer, X, kmeans, labels = cluster_failures(messages, k=N_CLUSTERS)

    print(f"\n=== Clusters with k={N_CLUSTERS} ===")
    print_top_terms(vectorizer, kmeans)

    print("\nCluster vs true failure type (how many of each type per cluster):")
    print(pd.crosstab(pd.Series(labels, name="cluster"), true_types.rename("true type")))

    print("\nScores:")
    print(f"  Silhouette:   {silhouette_score(X, labels):.3f}   (separation of clusters, no labels needed)")
    print(f"  ARI:          {adjusted_rand_score(true_types, labels):.3f}   (match with true types, 1 = perfect, 0 = random)")
    print(f"  Homogeneity:  {homogeneity_score(true_types, labels):.3f}   (each cluster contains one type)")
    print(f"  Completeness: {completeness_score(true_types, labels):.3f}   (each type sits in one cluster)")

    # 3) Picture: reduce to 2 dimensions and plot
    xy = TruncatedSVD(n_components=2, random_state=SEED).fit_transform(X)
    rng = np.random.default_rng(SEED)
    xy = xy + rng.normal(0, 0.01, xy.shape)  # small jitter: identical messages would overlap

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    axes[0].scatter(xy[:, 0], xy[:, 1], c=labels, cmap="tab10", s=18, alpha=0.7)
    axes[0].set_title(f"KMeans clusters (k={N_CLUSTERS})")
    for failure_type in sorted(true_types.unique()):
        mask = (true_types == failure_type).to_numpy()
        axes[1].scatter(xy[mask, 0], xy[mask, 1], s=18, alpha=0.7, label=failure_type)
    axes[1].set_title("True failure types (not used for clustering)")
    axes[1].legend()
    fig.tight_layout()
    IMAGE_FILE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(IMAGE_FILE, dpi=150)
    print(f"\nSaved cluster plot to {IMAGE_FILE}")


if __name__ == "__main__":
    main()