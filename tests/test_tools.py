"""Quick checks that every tool works.  Run from the project root:
    python tests/test_tools.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import tools  # noqa: E402


def test_failed_tests_hide_the_labels():
    result = tools.get_failed_tests(3, limit=5)
    assert result["total_failures"] == 117
    assert result["shown"] == 5
    assert "failure_type" not in result["failures"][0]


def test_unknown_run_gives_an_error():
    assert "error" in tools.get_run_summary(99)


def test_summary_flags_the_incident_runs():
    assert tools.get_run_summary(3)["anomalous"] is True
    assert tools.get_run_summary(7)["anomalous"] is True
    assert tools.get_run_summary(1)["anomalous"] is False


def test_classifier_recognizes_out_of_range():
    result = tools.classify_failure("voltage 17.2V exceeds limit 16V", "fail")
    assert result["predicted_type"] == "out_of_range"


def test_retry_pass_points_toward_flaky():
    msg = "timeout waiting for CAN message 0x1A3 after 500ms"
    p_pass = tools.classify_failure(msg, "pass")["probabilities"]["flaky"]
    p_fail = tools.classify_failure(msg, "fail")["probabilities"]["flaky"]
    assert p_pass > p_fail


def test_cluster_sizes_add_up():
    result = tools.cluster_failures(3)
    assert sum(c["size"] for c in result["clusters"]) == result["total_failures"] == 117


def test_similar_failures_are_found():
    result = tools.find_similar_past_failures(
        "timeout waiting for CAN message 0x1A3 after 500ms", exclude_run_id=3)
    assert result["n_matches"] > 0
    assert 3 not in result["runs_seen_in"]
    
def test_cluster_type_mix_adds_up():
    for cluster in tools.cluster_failures(3)["clusters"]:
        assert abs(sum(cluster["predicted_type_mix"].values()) - 1) < 0.05

if __name__ == "__main__":
    all_tests = [fn for name, fn in sorted(globals().items()) if name.startswith("test_")]
    for fn in all_tests:
        fn()
        print(f"PASS  {fn.__name__}")
    print(f"\nAll {len(all_tests)} checks passed.")