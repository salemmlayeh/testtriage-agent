"""Step 5: the tools an agent can call.

Every function takes simple inputs and returns a plain dict (JSON-friendly).
Run a demo from the project root:  python src/tools.py
"""
import json
from functools import lru_cache

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from cluster_failures import N_CLUSTERS, vectorize
from cluster_failures import cluster_failures as fit_clusters
from train_classifier import DATA_FILE, load_failures
from train_classifier_retry import build_pipeline_with_retry

# A run is "abnormal" if its failure rate is 1.5x the typical run or more
ANOMALY_FACTOR = 1.5


# ---------- cached helpers (computed once, then reused) ----------

@lru_cache(maxsize=1)
def _all_tests():
    return pd.read_csv(DATA_FILE)


@lru_cache(maxsize=1)
def _classifier():
    """Classifier from Step 3b (text + retry result), trained on all failures."""
    failures = load_failures()
    model = build_pipeline_with_retry()
    model.fit(failures[["error_message", "retry_result"]], failures["failure_type"])
    return model


@lru_cache(maxsize=1)
def _cluster_info():
    """Cluster all failures once, so cluster ids are the same for every run."""
    failures = load_failures()
    vectorizer, _, kmeans, labels = fit_clusters(failures["error_message"], k=N_CLUSTERS)
    terms = np.array(vectorizer.get_feature_names_out())
    top_terms = {
        i: [str(t) for t in terms[center.argsort()[::-1][:5]]]
        for i, center in enumerate(kmeans.cluster_centers_)
    }
    return labels, top_terms


@lru_cache(maxsize=1)
def _enriched_failures():
    """All failures plus the model's predicted type and the cluster id."""
    failures = load_failures().copy()
    failures["predicted_type"] = _classifier().predict(failures[["error_message", "retry_result"]])
    labels, _ = _cluster_info()
    failures["cluster"] = labels
    return failures


@lru_cache(maxsize=1)
def _similarity_index():
    failures = load_failures()
    return vectorize(failures["error_message"])  # (vectorizer, X)


# ---------- the tools ----------

def get_failed_tests(run_id: int, limit: int = 50) -> dict:
    """List failed tests of a run (message, ECU, duration, retry result)."""
    df = _all_tests()
    if df[df["run_id"] == run_id].empty:
        return {"error": f"run {run_id} not found"}
    failed = df[(df["run_id"] == run_id) & (df["result"] == "fail")]
    columns = ["test_id", "ecu", "error_message", "duration_ms", "retry_result"]
    return {
        "run_id": run_id,
        "total_failures": int(len(failed)),
        "shown": int(min(limit, len(failed))),
        "failures": failed[columns].head(limit).to_dict("records"),
    }


def get_run_summary(run_id: int) -> dict:
    """Compare a run with the other runs: is its failure rate abnormal?"""
    df = _all_tests()
    run = df[df["run_id"] == run_id]
    if run.empty:
        return {"error": f"run {run_id} not found",
                "available_runs": sorted(int(r) for r in df["run_id"].unique())}

    rate_by_run = (df["result"] == "fail").groupby(df["run_id"]).mean()
    run_rate = float(rate_by_run[run_id])
    baseline = float(rate_by_run.drop(run_id).median())  # typical rate of the other runs

    enriched = _enriched_failures()
    run_f = enriched[enriched["run_id"] == run_id]
    other_f = enriched[enriched["run_id"] != run_id]

    def mix(frame):
        shares = frame["predicted_type"].value_counts(normalize=True)
        return {str(k): round(float(v), 2) for k, v in shares.items()}

    retry_pass = float((run_f["retry_result"] == "pass").mean()) if len(run_f) else 0.0
    return {
        "run_id": run_id,
        "tests": int(len(run)),
        "failures": int((run["result"] == "fail").sum()),
        "fail_rate": round(run_rate, 3),
        "baseline_fail_rate": round(baseline, 3),
        "times_baseline": round(run_rate / baseline, 1),
        "anomalous": run_rate > ANOMALY_FACTOR * baseline,
        "predicted_type_mix": mix(run_f),
        "baseline_type_mix": mix(other_f),
        "top_ecus": run_f["ecu"].value_counts().head(3).to_dict(),
        "share_passed_on_retry": round(retry_pass, 2),
    }


def classify_failure(message: str, retry_result: str = "unknown") -> dict:
    """Predict the failure type of one error message (retry_result: 'pass', 'fail' or 'unknown')."""
    X = pd.DataFrame({"error_message": [message], "retry_result": [retry_result]})
    model = _classifier()
    proba = model.predict_proba(X)[0]
    probs = {str(c): round(float(p), 3) for c, p in zip(model.classes_, proba)}
    best = max(probs, key=probs.get)
    return {"predicted_type": best, "confidence": probs[best], "probabilities": probs}


def cluster_failures(run_id: int) -> dict:
    """Group a run's failures by message pattern, with retry statistics per group."""
    enriched = _enriched_failures()
    _, top_terms = _cluster_info()
    run_f = enriched[enriched["run_id"] == run_id]
    if run_f.empty:
        return {"run_id": run_id, "total_failures": 0, "clusters": []}

    clusters = []
    for cluster_id, group in run_f.groupby("cluster"):
        clusters.append({
            "cluster_id": int(cluster_id),
            "size": int(len(group)),
            "share_of_run": round(len(group) / len(run_f), 2),
            "top_terms": top_terms[int(cluster_id)],
            "predicted_type_mix": {
                str(k): round(float(v), 2)
                for k, v in group["predicted_type"].value_counts(normalize=True).items()
            },
            "example_per_type": group.groupby("predicted_type")["error_message"].first().to_dict(),
            "share_passed_on_retry": round(float((group["retry_result"] == "pass").mean()), 2),
            "affected_ecus": group["ecu"].value_counts().head(3).to_dict(),
        })
    clusters.sort(key=lambda c: c["size"], reverse=True)
    return {"run_id": run_id, "total_failures": int(len(run_f)), "clusters": clusters}


def find_similar_past_failures(message: str, min_similarity: float = 0.8,
                               exclude_run_id: int | None = None) -> dict:
    """Search past failures with a similar message.

    'known_causes' plays the role of resolved tickets in a knowledge base.
    """
    failures = load_failures()
    vectorizer, X = _similarity_index()
    sims = cosine_similarity(vectorizer.transform([message]), X).ravel()
    mask = sims >= min_similarity
    if exclude_run_id is not None:
        mask &= (failures["run_id"] != exclude_run_id).to_numpy()
    matches = failures[mask]
    if matches.empty:
        return {"n_matches": 0, "message": "no similar past failure found"}
    return {
        "n_matches": int(len(matches)),
        "known_causes": matches["failure_type"].value_counts().to_dict(),
        "share_passed_on_retry": round(float((matches["retry_result"] == "pass").mean()), 2),
        "runs_seen_in": sorted(int(r) for r in matches["run_id"].unique()),
        "examples": matches[["run_id", "test_id", "error_message", "retry_result"]]
                    .head(3).to_dict("records"),
    }


# The list we will hand to the agent in Step 6
TOOLS = [get_run_summary, get_failed_tests, cluster_failures,
         classify_failure, find_similar_past_failures]


if __name__ == "__main__":
    print("=== get_run_summary(3) ===")
    print(json.dumps(get_run_summary(3), indent=2))
    print("\n=== cluster_failures(3) ===")
    print(json.dumps(cluster_failures(3), indent=2))
    print("\n=== find_similar_past_failures(...) ===")
    print(json.dumps(find_similar_past_failures(
        "timeout waiting for CAN message 0x1A3 after 500ms", exclude_run_id=3), indent=2))