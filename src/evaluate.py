"""Step 9: evaluate the agent on questions whose answers are computed from the data.

Run from the project root:
    python src/evaluate.py --show-truth                  (no model calls)
    python src/evaluate.py --mode tools --limit 4        (agent with tools)
    python src/evaluate.py --mode notools --limit 6      (same model, no tools)
    python src/evaluate.py --report                      (table + docs/evaluation.md)
Every answer is saved right away, so you can continue on another day.
"""
import json
import os
import re
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "data" / "test_results.csv"
RESULTS_DIR = ROOT / "results"
REPORT_FILE = ROOT / "docs" / "evaluation.md"

load_dotenv()

ECUS = ["WindowLifter", "DoorLock", "BatteryMgmt", "SeatControl", "Wiper", "Lighting"]
TYPES = ["communication", "out_of_range", "timing", "flaky"]

QA_PROMPT = """You answer questions about ECU test runs. Use the tools and only numbers
from tool results. Reply with ONLY the final answer in the requested format, with no
explanation. If a run does not exist, reply NOT FOUND.
- A run is anomalous only if get_run_summary says anomalous is true.
- An ECU stands out only if it is listed in ecus_well_above_average.
- Failure counts per type are in predicted_type_counts.
- To compare runs or list anomalous runs, call compare_runs."""


NOTOOLS_PROMPT = """Answer the question about ECU test runs. Reply with ONLY the final
answer in the requested format, with no explanation."""


# ---------- checkers: each returns True if the answer is correct ----------

def ints_in(text):
    return [int(x) for x in re.findall(r"\d+", text)]


def check_int(expected):
    return lambda a: expected in ints_in(a)


def check_int_set(expected):
    return lambda a: set(ints_in(a)) == set(expected)


def check_pct(expected, tol=2):
    return lambda a: any(abs(float(x) - expected) <= tol
                         for x in re.findall(r"\d+(?:\.\d+)?", a))


def check_yes_no(expected):
    def check(a):
        m = re.match(r"\s*(yes|no)\b", a.lower())
        return bool(m) and m.group(1) == expected
    return check


def check_word(expected, options):
    def check(a):
        a = a.lower().replace(" ", "_")
        return expected.lower() in a and not any(
            o.lower() in a for o in options if o != expected)
    return check


def check_not_found(a):
    return bool(re.search(r"not found|does not exist|doesn't exist|no such run|not exist", a.lower()))


# ---------- questions, with answers computed from the CSV ----------

def build_questions():
    df = pd.read_csv(DATA_FILE)
    failed = df[df["result"] == "fail"]
    rate = (df["result"] == "fail").groupby(df["run_id"]).mean()
    n_fail = failed.groupby("run_id").size()
    anomalous = {int(r) for r in rate.index if rate[r] > 1.5 * rate.drop(r).median()}

    run3, run7 = failed[failed["run_id"] == 3], failed[failed["run_id"] == 7]
    retry_pass_3 = round(100 * float((run3["retry_result"] == "pass").mean()))
    top_ecu_3 = run3["ecu"].value_counts().idxmax()
    top_type_3 = run3["failure_type"].value_counts().idxmax()
    top_type_7 = run7["failure_type"].value_counts().idxmax()
    run7_all = df[df["run_id"] == 7]
    ecu_rate_7 = (run7_all["result"] == "fail").groupby(run7_all["ecu"]).mean()
    stands_out_7 = bool((ecu_rate_7 >= 1.5 * rate[7]).any())

    def q(i, text, fmt, expected, check):
        return {"id": i, "text": f"{text} {fmt}", "expected": str(expected), "check": check}

    return [
        q(1, "How many tests failed in run 3?", "Answer with a single integer.",
          n_fail[3], check_int(int(n_fail[3]))),
        q(2, "How many tests failed in run 7?", "Answer with a single integer.",
          n_fail[7], check_int(int(n_fail[7]))),
        q(3, "Which run has the highest failure rate?", "Answer with the run number only.",
          int(rate.idxmax()), check_int(int(rate.idxmax()))),
        q(4, "Which runs are anomalous?", "Answer with the run numbers only, comma separated.",
          sorted(anomalous), check_int_set(anomalous)),
        q(5, "Is run 1 anomalous?", "Answer yes or no.",
          "yes" if 1 in anomalous else "no", check_yes_no("yes" if 1 in anomalous else "no")),
        q(6, "Is run 7 anomalous?", "Answer yes or no.",
          "yes" if 7 in anomalous else "no", check_yes_no("yes" if 7 in anomalous else "no")),
        q(7, "What percentage of the failures in run 3 passed on retry?",
          "Answer with a whole number, no % sign.", retry_pass_3, check_pct(retry_pass_3)),
        q(8, "Which ECU had the most failures in run 3?", "Answer with the ECU name only.",
          top_ecu_3, check_word(top_ecu_3, ECUS)),
        q(9, "What is the most common failure type in run 3?",
          "Answer with one of: communication, out_of_range, timing, flaky.",
          top_type_3, check_word(top_type_3, TYPES)),
        q(10, "What is the most common failure type in run 7?",
          "Answer with one of: communication, out_of_range, timing, flaky.",
          top_type_7, check_word(top_type_7, TYPES)),
        q(11, "Does any single ECU stand out in run 7?", "Answer yes or no.",
          "yes" if stands_out_7 else "no", check_yes_no("yes" if stands_out_7 else "no")),
        q(12, "How many tests failed in run 99?",
          "Answer with a single integer, or NOT FOUND if run 99 does not exist.",
          "NOT FOUND", check_not_found),
    ]


# ---------- running and saving ----------

def load_records(mode, model_name):
    path = RESULTS_DIR / f"eval_{mode}.jsonl"
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    records = [json.loads(line) for line in lines if line.strip()]
    return [r for r in records if r["model"] == model_name]


def append_record(mode, record):
    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / f"eval_{mode}.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def ask_with_tools(agent, question):
    result = agent.invoke({"messages": [{"role": "user", "content": question}]})
    msgs = result["messages"]
    tool_calls = sum(len(getattr(m, "tool_calls", None) or []) for m in msgs)
    model_calls = sum(type(m).__name__ == "AIMessage" for m in msgs)
    return msgs[-1].text.strip(), tool_calls, model_calls


def ask_without_tools(model, question):
    answer = model.invoke([("system", NOTOOLS_PROMPT), ("user", question)]).text.strip()
    return answer, 0, 1


def short(text, n=40):
    return text.replace("\n", " ").replace("|", "/")[:n]


def write_report(model_name, questions):
    by_mode = {m: {r["id"]: r for r in load_records(m, model_name)} for m in ("tools", "notools")}
    lines = ["# Evaluation results", "",
             f"Model: `{model_name}` (temperature 0). Answer key computed from the CSV.", "",
             "| # | Question | Expected | With tools | Without tools |", "|---|---|---|---|---|"]
    for q in questions:
        cells = []
        for mode in ("tools", "notools"):
            r = by_mode[mode].get(q["id"])
            cells.append("not run" if r is None else
                         f"{'✅' if r['correct'] else '❌'} {short(r['answer'])}")
        lines.append(f"| {q['id']} | {short(q['text'].split('?')[0] + '?', 60)} | "
                     f"{short(q['expected'], 20)} | {cells[0]} | {cells[1]} |")
    lines.append("")
    for mode, label in (("tools", "With tools"), ("notools", "Without tools")):
        recs = list(by_mode[mode].values())
        if recs:
            ok = sum(r["correct"] for r in recs)
            avg_tools = sum(r["tool_calls"] for r in recs) / len(recs)
            lines.append(f"- **{label}:** {ok}/{len(recs)} correct "
                         f"({100 * ok / len(recs):.0f}%), {avg_tools:.1f} tool calls per question")
    REPORT_FILE.parent.mkdir(exist_ok=True)
    REPORT_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nSaved to {REPORT_FILE}")


def main():
    args = sys.argv[1:]
    questions = build_questions()

    if "--show-truth" in args:
        for q in questions:
            print(f"{q['id']:>2}. {q['text']}\n    expected: {q['expected']}")
        return

    model_name = os.environ["LLM_MODEL"]
    if "--report" in args:
        write_report(model_name, questions)
        return

    mode = args[args.index("--mode") + 1] if "--mode" in args else "tools"
    limit = int(args[args.index("--limit") + 1]) if "--limit" in args else 4
    if mode not in ("tools", "notools"):
        sys.exit("--mode must be tools or notools")

    done = {r["id"] for r in load_records(mode, model_name)}
    todo = [q for q in questions if q["id"] not in done][:limit]
    if not todo:
        print(f"All questions already answered in mode '{mode}' for {model_name}.")
        return

    model = init_chat_model(model_name, temperature=0)
    qa_agent = None
    if mode == "tools":
        from langchain.agents import create_agent
        from agent import TOOLS
        qa_agent = create_agent(model, tools=TOOLS, system_prompt=QA_PROMPT)

    print(f"Mode: {mode} | model: {model_name} | questions this session: {len(todo)}\n")
    for q in todo:
        try:
            if mode == "tools":
                answer, tool_calls, model_calls = ask_with_tools(qa_agent, q["text"])
            else:
                answer, tool_calls, model_calls = ask_without_tools(model, q["text"])
        except Exception as exc:
            text = str(exc)
            if "429" in text or "RESOURCE_EXHAUSTED" in text:
                print("Rate limit reached. Progress is saved; run the same command later.")
            else:
                print(f"Failed: {text[:300]}")
            break
        correct = bool(q["check"](answer))
        append_record(mode, {"id": q["id"], "model": model_name, "mode": mode,
                             "question": q["text"], "expected": q["expected"],
                             "answer": answer, "correct": correct,
                             "tool_calls": tool_calls, "model_calls": model_calls})
        print(f"{q['id']:>2}. {'PASS' if correct else 'FAIL'}  answer: {short(answer, 50)!r}"
              f"  expected: {q['expected']}  (tool calls: {tool_calls})")

    recs = load_records(mode, model_name)
    print(f"\nProgress ({mode}): {len(recs)}/{len(questions)} answered, "
          f"{sum(r['correct'] for r in recs)} correct so far.")


if __name__ == "__main__":
    main()