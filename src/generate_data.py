"""Generate synthetic E/E test results for the TestTriage project.

All data is invented. Nothing here comes from a real company or vehicle.
Run from the project root:  python src/generate_data.py
"""
import random
from pathlib import Path

import pandas as pd

SEED = 42             # same seed = same data every time (reproducible)
N_RUNS = 10           # number of test runs
TESTS_PER_RUN = 300   # tests executed in each run

random.seed(SEED)

# Fake ECUs (electronic control units) that the tests belong to
ECUS = ["WindowLifter", "DoorLock", "BatteryMgmt", "SeatControl", "Wiper", "Lighting"]

# Error message templates for each failure type.
# Words in {braces} get replaced by random values below.
TEMPLATES = {
    "communication": [
        "timeout waiting for CAN message 0x{can_id:03X} after {wait_ms}ms",
        "CAN message 0x{can_id:03X} not received within {wait_ms}ms",
        "bus error: no response from node 0x{can_id:03X}",
    ],
    "out_of_range": [
        "voltage {volt}V exceeds limit {volt_limit}V",
        "signal value {value} out of range [{low}, {high}]",
        "current {amp}A above max {amp_limit}A",
    ],
    "timing": [
        "response took {took_ms}ms, max {limit_ms}ms",
        "latency {took_ms}ms exceeds allowed {limit_ms}ms",
        "cycle time violation: {took_ms}ms > {limit_ms}ms",
    ],
    "flaky": [
        "assertion failed: unexpected state transition",
        "intermittent mismatch in signal check",
        # Same text as a communication error on purpose: real logs are
        # ambiguous, so a text-only classifier will not be perfect.
        "timeout waiting for CAN message 0x{can_id:03X} after {wait_ms}ms",
    ],
}

# Normal mix of failure types, and normal failure rate
DEFAULT_WEIGHTS = {"communication": 0.30, "out_of_range": 0.25, "timing": 0.25, "flaky": 0.20}
DEFAULT_FAIL_RATE = 0.12

# Special runs with an "incident" for the agent to find later
RUN_PROFILES = {
    3: {"fail_rate": 0.35,
        "weights": {"communication": 0.70, "out_of_range": 0.10, "timing": 0.10, "flaky": 0.10}},
    7: {"fail_rate": 0.25,
        "weights": {"communication": 0.10, "out_of_range": 0.60, "timing": 0.15, "flaky": 0.15}},
}


def random_values():
    """Random numbers that can be inserted into any message template."""
    return {
        "can_id": random.randint(0x100, 0x7FF),
        "wait_ms": random.choice([100, 200, 500, 1000]),
        "volt": round(random.uniform(16.5, 19.5), 1),
        "volt_limit": 16,
        "amp": round(random.uniform(10.5, 20.0), 1),
        "amp_limit": 10,
        "value": round(random.uniform(5.5, 9.0), 1),
        "low": 0,
        "high": 5,
        "took_ms": random.randint(600, 1500),
        "limit_ms": 500,
    }


def make_failure(failure_type):
    """Return (error_message, duration_ms) for one failed test."""
    values = random_values()
    template = random.choice(TEMPLATES[failure_type])
    message = template.format(**values)  # unused keys are simply ignored

    if failure_type == "timing":
        duration = values["took_ms"]
    elif failure_type == "communication":
        duration = values["wait_ms"]
    else:
        duration = int(random.gauss(300, 50))
    return message, duration


def make_row(run_id, test_num, fail_rate, weights):
    """Create one row (one test execution) of the results table."""
    ecu = ECUS[(test_num - 1) % len(ECUS)]
    row = {
        "run_id": run_id,
        "test_id": f"Test_{ecu}_{test_num:03d}",
        "ecu": ecu,
    }

    if random.random() < fail_rate:
        failure_type = random.choices(list(weights), weights=list(weights.values()))[0]
        message, duration = make_failure(failure_type)
        row.update(
            result="fail",
            failure_type=failure_type,  # the hidden "ground truth" label
            error_message=message,
            duration_ms=duration,
            # flaky tests usually pass when retried, real failures do not
            retry_result="pass" if (failure_type == "flaky" and random.random() < 0.9) else "fail",
        )
    else:
        row.update(
            result="pass",
            failure_type=None,
            error_message=None,
            duration_ms=int(random.gauss(300, 50)),
            retry_result=None,
        )
    return row


def main():
    rows = []
    for run_id in range(1, N_RUNS + 1):
        profile = RUN_PROFILES.get(run_id, {})
        fail_rate = profile.get("fail_rate", DEFAULT_FAIL_RATE)
        weights = profile.get("weights", DEFAULT_WEIGHTS)
        for test_num in range(1, TESTS_PER_RUN + 1):
            rows.append(make_row(run_id, test_num, fail_rate, weights))

    df = pd.DataFrame(rows)

    out_dir = Path(__file__).resolve().parent.parent / "data"
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / "test_results.csv"
    df.to_csv(out_file, index=False)

    print(f"Saved {len(df)} rows to {out_file}\n")
    print("Failures per run:")
    print(df[df["result"] == "fail"].groupby("run_id").size(), "\n")
    print("Failure types overall:")
    print(df["failure_type"].value_counts())


if __name__ == "__main__":
    main()