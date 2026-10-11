import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pipeline  # noqa: E402


def test_parse_ticket():
    text = "TITLE: Bus fault in run 3\nSEVERITY: HIGH\nDESCRIPTION: 117 failures.\nCheck the bus."
    ticket = pipeline.parse_ticket(text)
    assert ticket["severity"] == "high"
    assert ticket["title"].startswith("Bus fault")
    assert "117" in ticket["description"]


def test_no_ticket_gives_none():
    assert pipeline.parse_ticket("NO_TICKET") is None


if __name__ == "__main__":
    for fn in (test_parse_ticket, test_no_ticket_gives_none):
        fn()
        print(f"PASS  {fn.__name__}")
