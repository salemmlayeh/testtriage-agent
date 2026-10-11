import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import guardrails  # noqa: E402
import tools  # noqa: E402


def test_flags_ecu_and_can_id():
    report = "VERDICT: x\nLIKELY CAUSES:\n1. DoorLock overcurrent seen at 0x636\nNEXT STEPS:\n- inspect Wiper"
    assert len(guardrails.check_report(report, 7)) >= 2


def test_accepts_clean_report():
    report = "VERDICT: anomalous.\nLIKELY CAUSES:\n1. Failures spread evenly across ECUs, suggesting a shared cause."
    assert guardrails.check_report(report, 7) == []


def test_flags_intermittent_on_persistent_cluster():
    clusters = tools.cluster_failures(7)["clusters"]
    persistent = [c["cluster_id"] for c in clusters if c["share_passed_on_retry"] < 0.1]
    assert persistent
    report = f"LIKELY CAUSES:\n1. Cluster {persistent[0]} shows intermittent timeouts."
    assert guardrails.check_report(report, 7)


if __name__ == "__main__":
    for fn in (test_flags_ecu_and_can_id, test_accepts_clean_report,
               test_flags_intermittent_on_persistent_cluster):
        fn()
        print(f"PASS  {fn.__name__}")