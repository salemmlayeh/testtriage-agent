# Evaluation results

Model: `google_genai:gemini-3.1-flash-lite` (temperature 0). Answer key computed from the CSV.

| # | Question | Expected | With tools | Without tools |
|---|---|---|---|---|
| 1 | How many tests failed in run 3? | 117 | ✅ 117 | ❌ 0 |
| 2 | How many tests failed in run 7? | 75 | ✅ 75 | ❌ 0 |
| 3 | Which run has the highest failure rate? | 3 | ✅ 3 | ❌ 4 |
| 4 | Which runs are anomalous? | [3, 7] | ❌ 3 | ❌ 12, 27, 44 |
| 5 | Is run 1 anomalous? | no | not run | ❌ Yes |
| 6 | Is run 7 anomalous? | yes | not run | ❌ No |
| 7 | What percentage of the failures in run 3 passed on retry? | 11 | not run | ❌ 50 |
| 8 | Which ECU had the most failures in run 3? | WindowLifter | not run | ❌ ECU-4 |
| 9 | What is the most common failure type in run 3? | communication | not run | ❌ timing |
| 10 | What is the most common failure type in run 7? | out_of_range | not run | ❌ timing |
| 11 | Does any single ECU stand out in run 7? | no | not run | ✅ No |
| 12 | How many tests failed in run 99? | NOT FOUND | not run | ✅ NOT FOUND |

- **With tools:** 3/4 correct (75%), 2.0 tool calls per question
- **Without tools:** 2/12 correct (17%), 0.0 tool calls per question
