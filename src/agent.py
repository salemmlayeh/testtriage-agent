"""Step 6: the TestTriage agent.

Run from the project root:
    python src/agent.py "Analyze test run 3"
"""
import os
import sys

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model

import tools as t

load_dotenv()

SYSTEM_PROMPT = """You are a test triage assistant for electronic control unit (ECU) test runs.
You help an engineer understand why tests failed. You have tools; use them.

Work like this:
1. Call get_run_summary first. If the run is not anomalous, say so plainly and stop
   digging; do not invent a problem.
2. If it is anomalous, call cluster_failures to see which failures belong together.
3. For any cluster whose predicted_type_mix is mixed, or whose share_passed_on_retry
   is between 0.05 and 0.95, call find_similar_past_failures with an example message
   and the run id you are analyzing as exclude_run_id.
4. Write the report.

Rules:
- Use only numbers that appear in tool results. Never invent or estimate figures.
- Failures that pass on retry point to flaky tests; failures that fail again point to
  real faults.
- Causes are hypotheses, not facts. Say "suggests" or "is consistent with".
- Failures spread over many ECUs suggest a shared cause (for example the bus),
  while failures on one ECU suggest a problem in that ECU.
- If a tool returns an error, report it instead of guessing.
- A cluster groups similar wording, not one failure type. If predicted_type_mix has
  several types, say so. Never call a mixed cluster by a single type.
- Count failures per type from get_run_summary's predicted_type_counts, not from
  cluster sizes.
- Describe a cluster only with facts from top_terms and example_per_type. Do not
  generalize one example (a single voltage or current value) to the whole cluster.
- Only name ECUs that appear in tool results.
- A cluster with share_passed_on_retry below 0.1 is persistent, not intermittent.
- Name an ECU as a suspect only if it is listed in ecus_well_above_average. If that
  list is empty, say failures are spread evenly across ECUs and do not recommend
  inspecting any specific ECU.
- Never mention individual CAN ids (like 0x636) in the report; they differ in every
  message and one example says nothing about a faulty node.
- For questions about several runs (which runs are anomalous, which run is worst),
  call compare_runs instead of checking runs one by one.
Report format (plain text, under 250 words):
VERDICT: one sentence.
EVIDENCE: 3-5 short lines with the key numbers.
LIKELY CAUSES: ranked hypotheses.
FLAKY NOISE: how much of the run looks flaky.
NEXT STEPS: 2-3 concrete actions for the engineer.
"""


def find_similar_past_failures(message: str, exclude_run_id: int) -> dict:
    """Search past failures with a similar error message and report their known causes.

    Pass the id of the run you are analyzing as exclude_run_id, so it is left out.
    """
    return t.find_similar_past_failures(message, exclude_run_id=exclude_run_id)


TOOLS = [
    t.get_run_summary,
    t.get_failed_tests,
    t.cluster_failures,
    t.classify_failure,
    find_similar_past_failures,
    t.compare_runs,
]


def print_trace(messages):
    """Show which tools the agent called, in order, and how many model calls it needed."""
    print("--- Tool calls ---")
    n_calls = 0
    n_model_calls = 0
    for m in messages:
        if type(m).__name__ == "AIMessage":
            n_model_calls += 1
        for call in getattr(m, "tool_calls", None) or []:
            n_calls += 1
            print(f"{n_calls}. {call['name']}({call['args']})")
    print(f"(model calls: {n_model_calls})\n")


def main():
    question = " ".join(sys.argv[1:]) or "Analyze test run 3 and tell me what is going wrong."
    model = init_chat_model(os.environ["LLM_MODEL"], temperature=0)
    agent = create_agent(model, tools=TOOLS, system_prompt=SYSTEM_PROMPT)

    print(f"Question: {question}\n")
    try:
        result = agent.invoke({"messages": [{"role": "user", "content": question}]})
    except Exception as exc:  # most likely a free-tier rate limit
        text = str(exc)
        if "429" in text or "RESOURCE_EXHAUSTED" in text:
            print("Rate limit reached on the free tier. Wait a minute and try again,")
            print("or switch LLM_MODEL in .env to another free model.")
        else:
            print(f"Agent failed: {text}")
        sys.exit(1)

    print_trace(result["messages"])
    print("--- Report ---")
    print(result["messages"][-1].text)


if __name__ == "__main__":
    main()