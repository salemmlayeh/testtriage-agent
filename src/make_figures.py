"""Extra figures for the README. No model calls, no API key needed.

Run from the project root:  python src/make_figures.py
Writes PNG files to docs/images/.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "data" / "test_results.csv"
RESULTS_DIR = ROOT / "results"
IMG_DIR = ROOT / "docs" / "images"

TYPES = ["communication", "out_of_range", "timing", "flaky"]
COLORS = {"communication": "#1f77b4", "out_of_range": "#2ca02c",
          "timing": "#d62728", "flaky": "#ff7f0e"}
ANOMALY_FACTOR = 1.5  # same rule as src/tools.py


def save(fig, name):
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    path = IMG_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def figure_failures_per_run(df):
    """Bar chart of failures per run; anomalous runs are highlighted."""
    fails = df[df["result"] == "fail"].groupby("run_id").size()
    rate = (df["result"] == "fail").groupby(df["run_id"]).mean()
    anomalous = {r for r in rate.index if rate[r] > ANOMALY_FACTOR * rate.drop(r).median()}
    tests_per_run = int(df.groupby("run_id").size().median())

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar([str(r) for r in fails.index], fails.values,
                  color=["#d62728" if r in anomalous else "#8c8c8c" for r in fails.index])
    ax.bar_label(bars)
    threshold = ANOMALY_FACTOR * float(fails.median())
    ax.axhline(threshold, color="black", linestyle="--", linewidth=1)
    ax.text(len(fails) - 0.55, threshold + 2, "1.5 x median run", ha="right", fontsize=8)
    ax.set_xlabel("Run")
    ax.set_ylabel("Failed tests")
    ax.set_title(f"Failures per run ({tests_per_run} tests each)")
    ax.legend(handles=[Patch(color="#d62728", label="anomalous run"),
                       Patch(color="#8c8c8c", label="normal run")], loc="upper right")
    save(fig, "09_failures_per_run.png")


def figure_failure_types_per_run(df):
    """Stacked bars of failure types per run (true labels from the generator)."""
    mix = (df[df["result"] == "fail"].groupby(["run_id", "failure_type"]).size()
           .unstack(fill_value=0).reindex(columns=TYPES, fill_value=0))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    mix.plot(kind="bar", stacked=True, ax=ax, color=[COLORS[t] for t in TYPES], width=0.8)
    ax.set_xlabel("Run")
    ax.set_ylabel("Failed tests")
    ax.set_xticklabels([str(r) for r in mix.index], rotation=0)
    ax.set_title("Failure types per run (true labels from the generator)")
    ax.legend(title="Failure type")
    save(fig, "10_failure_types_per_run.png")


def figure_ecu_rates(df, runs=(3, 7)):
    """Failure rate per ECU, with the 'stands out' threshold used by the tools."""
    fig, axes = plt.subplots(1, len(runs), figsize=(11, 4.5), sharex=True)
    for ax, run_id in zip(np.atleast_1d(axes), runs):
        run = df[df["run_id"] == run_id]
        overall = float((run["result"] == "fail").mean())
        by_ecu = (run["result"] == "fail").groupby(run["ecu"]).mean().sort_values()
        bars = ax.barh(by_ecu.index, by_ecu.values, color="#6e6e6e")
        ax.bar_label(bars, fmt="%.2f", label_type="center", color="white")
        ax.axvline(overall, color="black", linewidth=1, label=f"run failure rate ({overall:.2f})")
        ax.axvline(ANOMALY_FACTOR * overall, color="#d62728", linestyle="--",
                   label=f"1.5 x run rate ({ANOMALY_FACTOR * overall:.2f}): ECU stands out")
        ax.set_title(f"Run {run_id}: failure rate per ECU")
        ax.set_xlabel("Failure rate")
        ax.set_xlim(0, 0.75)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), fontsize=8, frameon=False)
    save(fig, "11_ecu_rates.png")


def load_answers(mode):
    """Latest correct/incorrect flag per question id, from results/eval_<mode>.jsonl."""
    path = RESULTS_DIR / f"eval_{mode}.jsonl"
    answers = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                answers[record["id"]] = bool(record["correct"])
    return answers


def figure_evaluation():
    """Pass/fail grid: one cell per question, with and without tools."""
    with_tools, without_tools = load_answers("tools"), load_answers("notools")
    if not with_tools and not without_tools:
        print("No evaluation results found in results/, skipping the evaluation figure.")
        return
    n_questions = max(list(with_tools) + list(without_tools))
    grid = np.full((2, n_questions), np.nan)  # nan = not run, 0 = wrong, 1 = right
    for row, answers in enumerate((with_tools, without_tools)):
        for qid, ok in answers.items():
            grid[row, qid - 1] = 1.0 if ok else 0.0

    fig, ax = plt.subplots(figsize=(10, 2.8))
    cmap = ListedColormap(["#e6a1a1", "#a6d8a8"])
    ax.imshow(np.ma.masked_invalid(grid), cmap=cmap, vmin=0, vmax=1, aspect="auto")
    for row in range(2):
        for col in range(n_questions):
            value = grid[row, col]
            symbol = "-" if np.isnan(value) else ("\u2713" if value == 1 else "\u2717")
            ax.text(col, row, symbol, ha="center", va="center", fontsize=14)
    ax.set_xticks(range(n_questions))
    ax.set_xticklabels([f"Q{i + 1}" for i in range(n_questions)])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["With tools", "Without tools"])
    ax.set_xticks(np.arange(-0.5, n_questions), minor=True)
    ax.set_yticks(np.arange(-0.5, 2), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="both", length=0)

    def score(answers):
        return f"{sum(answers.values())}/{len(answers)}"

    ax.set_title(f"Evaluation: {score(with_tools)} correct with tools, "
                 f"{score(without_tools)} without tools")
    save(fig, "12_evaluation_grid.png")


def main():
    if not DATA_FILE.exists():
        raise SystemExit("data/test_results.csv not found. Run: python src/generate_data.py")
    df = pd.read_csv(DATA_FILE)
    figure_failures_per_run(df)
    figure_failure_types_per_run(df)
    figure_ecu_rates(df)
    figure_evaluation()


if __name__ == "__main__":
    main()