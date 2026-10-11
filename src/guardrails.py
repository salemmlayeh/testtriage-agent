"""Rule-based checks on an agent report. No model calls, so they cost no quota."""
import re

import tools

ECUS = ["WindowLifter", "DoorLock", "BatteryMgmt", "SeatControl", "Wiper", "Lighting"]
CAN_ID = re.compile(r"\b0x[0-9A-Fa-f]{2,3}\b")


def check_report(report: str, run_id: int) -> list[str]:
    """Return a list of problems found in the report (empty list = no problems)."""
    problems = []
    summary = tools.get_run_summary(run_id)
    clusters = tools.cluster_failures(run_id).get("clusters", [])

    # 1) ECUs named as suspects although no ECU stands out
    causes_and_steps = report.split("LIKELY CAUSES", 1)[1] if "LIKELY CAUSES" in report else ""
    if not summary.get("ecus_well_above_average"):
        named = [e for e in ECUS if e.lower() in causes_and_steps.lower()]
        if named:
            problems.append(
                f"The report names {', '.join(named)} in the causes or next steps, but no ECU "
                "has a failure rate well above average. Say failures are spread evenly across "
                "ECUs and do not name any ECU as a suspect."
            )

    # 2) CAN ids taken from single example messages
    ids = sorted(set(CAN_ID.findall(report)))
    if ids:
        problems.append(
            f"The report mentions CAN ids ({', '.join(ids)}). Every message has a different id, "
            "so remove them."
        )

    # 3) a persistent cluster described as intermittent
    persistent = {c["cluster_id"] for c in clusters if c["share_passed_on_retry"] < 0.1}
    for line in report.splitlines():
        if re.search(r"intermittent|transient|flaky", line, re.I):
            ids_in_line = re.findall(r"[Cc]luster\s*(\d+)", line)
            if len(ids_in_line) == 1 and int(ids_in_line[0]) in persistent:
                problems.append(
                    f"Cluster {ids_in_line[0]} passed on retry in under 10% of cases, so it is "
                    "persistent, not intermittent or flaky."
                )
    return problems