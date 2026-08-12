"""Verify the hash chain of every committed ledger file. CI helper.

Usage: python scripts/verify_ledgers.py [ROOT]

Walks ROOT (default: experiments/) for *.jsonl files and runs
ExperimentLedger.verify() on each. Exits non-zero if any chain is broken.
"""

import sys
from pathlib import Path

from integrity_harness.ledger import ExperimentLedger


def main(root: str = "experiments") -> int:
    paths = sorted(Path(root).rglob("*.jsonl"))
    if not paths:
        print(f"No ledger files found under {root}/")
        return 0
    failed = False
    for p in paths:
        try:
            verified = ExperimentLedger(p).verify()
        except ValueError as e:
            print(f"::error::{e}")
            failed = True
            continue
        print(f"OK: {p} ({verified} chained records verified)")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:2]))
