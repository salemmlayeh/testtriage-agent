"""Step 7: analysis + self-check (reflection) + human approval.

Run from the project root:
    python src/pipeline.py "Analyze test run 3"
    python src/pipeline.py "Analyze test run 3" --no-review --no-ticket   (saves model calls)
"""
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model

from agent import SYSTEM_PROMPT, TOOLS, print_trace

load_dotenv()
TICKET_DIR = Path(__file__).resolve().parent.parent / "tickets"

REVIEW_PROMPT = """You are a strict fact-checker for test triage reports.
You get TOOL RESULTS (the ground truth) and a REPORT written from them.
List every claim in the REPORT that is not supported by the TOOL RESULTS or that
contradicts them. Check numbers, cluster descriptions, ECU names, and words like
"intermittent" (a cluster with share_passed_on_retry below 0.1 is persistent).
Also flag recommendations that contradict the evidence, for example inspecting an
ECU when the report says failures are spread evenly.
Answer with a numbered list, one line per problem. If everything is supported,
answer exactly: OK"""

REVISE_PROMPT = """Rewrite the REPORT so that it fixes every listed PROBLEM.
Keep the same format and stay under 250 words. Use only facts from the TOOL RESULTS.
Return only the corrected report."""

TICKET_PROMPT = """Draft a defect ticket from the FINAL REPORT.
If the report says the run is not anomalous, answer exactly: NO_TICKET
Otherwise answer in exactly this format:
TITLE: <one line, max 90 characters>
SEVERITY: <low|medium|high>
DESCRIPTION: <3 to 6 lines: what happened, key numbers, suspected causes as
hypotheses, suggested next steps>
Use only facts from the report."""


def collect_tool_results(messages):
    """Gather everything the tools returned, as plain text."""
    parts = [f"[{m.name}] {m.content}" for m in messages if type(m).__name__ == "ToolMessage"]
    return "\n".join(parts)


def ask(model, prompt):
    return model.invoke(prompt).text.strip()


def review_report(model, report, tool_results):
    return ask(model, f"{REVIEW_PROMPT}\n\nTOOL RESULTS:\n{tool_results}\n\nREPORT:\n{report}")


def revise_report(model, report, problems, tool_results):
    return ask(model, f"{REVISE_PROMPT}\n\nTOOL RESULTS:\n{tool_results}\n\n"
                      f"REPORT:\n{report}\n\nPROBLEMS:\n{problems}")


def parse_ticket(text):
    """Turn the model's text into a dict, or None if there is no ticket."""
    text = text.replace("*", "")
    title = re.search(r"TITLE:\s*(.+)", text)
    severity = re.search(r"SEVERITY:\s*(\w+)", text)
    description = re.search(r"DESCRIPTION:\s*(.+)", text, re.S)
    if not (title and severity and description):
        return None
    return {"title": title.group(1).strip(),
            "severity": severity.group(1).strip().lower(),
            "description": description.group(1).strip()}


def approve_and_save(ticket, run_label):
    """Human in the loop: nothing is created unless the engineer types y."""
    print("--- Proposed defect ticket ---")
    print(f"Title:    {ticket['title']}")
    print(f"Severity: {ticket['severity']}")
    print(f"{ticket['description']}\n", flush=True)
    try:
        answer = input("Create this ticket? [y/N] ").strip().lower()
    except EOFError:
        answer = ""
    if answer != "y":
        print("Ticket NOT created (rejected by the engineer).")
        return
    TICKET_DIR.mkdir(exist_ok=True)
    path = TICKET_DIR / f"ticket_run{run_label}.md"
    path.write_text(f"# {ticket['title']}\n\nSeverity: {ticket['severity']}\n\n"
                    f"{ticket['description']}\n", encoding="utf-8")
    print(f"Ticket saved to {path}")


def main():
    args = sys.argv[1:]
    flags = {a for a in args if a.startswith("--")}
    question = " ".join(a for a in args if not a.startswith("--")) \
        or "Analyze test run 3 and tell me what is going wrong."
    match = re.search(r"run\s*(\d+)", question, re.I)
    run_label = match.group(1) if match else "x"

    model = init_chat_model(os.environ["LLM_MODEL"], temperature=0)
    agent = create_agent(model, tools=TOOLS, system_prompt=SYSTEM_PROMPT)

    try:
        print(f"Question: {question}\n")
        result = agent.invoke({"messages": [{"role": "user", "content": question}]})
        messages = result["messages"]
        print_trace(messages)
        report = messages[-1].text
        tool_results = collect_tool_results(messages)
        print("--- Draft report ---")
        print(report)

        if "--no-review" not in flags:
            print("\n--- Self-check ---")
            problems = review_report(model, report, tool_results)
            if problems.strip().strip(".").upper() == "OK":
                print("Reviewer found no unsupported claims.")
            else:
                print(problems)
                report = revise_report(model, report, problems, tool_results)
                print("\n--- Revised report ---")
                print(report)

        if "--no-ticket" not in flags:
            print()
            draft = ask(model, f"{TICKET_PROMPT}\n\nFINAL REPORT:\n{report}")
            ticket = parse_ticket(draft)
            if ticket is None:
                print("No ticket needed for this run.")
            else:
                approve_and_save(ticket, run_label)
    except Exception as exc:
        text = str(exc)
        if "429" in text or "RESOURCE_EXHAUSTED" in text:
            print(text[:600])
            print("Rate limit reached on the free tier. Wait a minute and try again,")
            print("or add --no-review --no-ticket to use fewer model calls.")
        else:
            print(f"Failed: {text}")
        sys.exit(1)


if __name__ == "__main__":
    main()