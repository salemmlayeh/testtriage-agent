# Agent notes (findings during development)

1. First run 7 report called a mixed cluster (out_of_range + timing) by one type.
   Fix: tools now return predicted_type_mix, predicted_type_counts and example_per_type.
2. Agent over-read ECU noise (6 ECUs, ~12 failures each). Fix: failure_rate_by_ecu plus
   ecus_well_above_average (1.5x rule) computed in code, not by the model.
3. Prompt rules alone did not stop contradictions (ECU named as suspect while saying
   failures are spread evenly). Fix: rule-based guardrails in src/guardrails.py.
4. LLM self-check made run 7 worse: reviewer ignored the 1.5x rule and the revision named
   DoorLock. Guardrails caught it after revision.
   Takeaway: use deterministic checks first, an LLM reviewer second, and keep the draft
   if the revision fails the rules.
5. Free-tier quota: gemini-3-flash-preview allowed 20 requests/day. Switched to
   gemini-3.1-flash-lite; evaluation needs many calls, so quota planning matters.

6. Evaluation (12 questions, answers computed from the CSV): without tools the model got
   2/12, and both hits were chance (a yes/no guess and "NOT FOUND", which it answers for
   any run). With tools (v1) it failed "which runs are anomalous": every tool worked on one
   run, so the agent checked only some runs. Fix: added a compare_runs tool (v2).

7. Final evaluation (v2, gemini-3.1-flash-lite): 11/12 with tools, 2/12 without. The miss was
   "which ECU had the most failures in run 3": the agent answered Wiper (22) instead of
   WindowLifter (24), although get_run_summary lists top_ecus directly. Probable cause: it
   aggregated cluster-level affected_ecus (top 3 per cluster only). Not tuned away on purpose,
   to keep the test independent. Question 4 was fixed after a failure (compare_runs), so it is
   not an independent test either.
