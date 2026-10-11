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
