"""Tests for the ledger's tamper-evident hash chain.

Each record carries prev_hash = SHA-256 of the previous record's raw line
bytes (genesis records use a fixed all-zeros sentinel). verify() walks the
chain and raises on any edit, insertion, or deletion of a chained record.
Records written before hash-chaining existed (no prev_hash field) are
tolerated as a legacy prefix.
"""

import hashlib
import json
from pathlib import Path

import pytest

from integrity_harness.ledger import GENESIS_HASH, ExperimentLedger


@pytest.fixture()
def chained(tmp_path):
    path = tmp_path / "ledger.jsonl"
    led = ExperimentLedger(path)
    led.record({"a": 1}, 0.1, 100)
    led.record({"a": 2}, 0.2, 100)
    led.record({"a": 3}, 0.3, 100)
    return led, path


def test_chain_worked_example(chained):
    # Hand-computed check: prev_hash of record k is the SHA-256 of the raw
    # bytes of line k-1, and the genesis record uses the sentinel.
    led, path = chained
    lines = path.read_bytes().splitlines()
    recs = [json.loads(ln) for ln in lines]
    assert recs[0]["prev_hash"] == GENESIS_HASH == "0" * 64
    assert recs[1]["prev_hash"] == hashlib.sha256(lines[0]).hexdigest()
    assert recs[2]["prev_hash"] == hashlib.sha256(lines[1]).hexdigest()
    assert led.verify() == 3


def test_verify_detects_edit(chained):
    led, path = chained
    text = path.read_text().replace('"sharpe": 0.2', '"sharpe": 2.0')
    path.write_text(text)
    with pytest.raises(ValueError, match="hash chain"):
        led.verify()


def test_verify_detects_deleted_record(chained):
    led, path = chained
    lines = path.read_text().splitlines(keepends=True)
    path.write_text(lines[0] + lines[2])  # drop the middle record
    with pytest.raises(ValueError, match="hash chain"):
        led.verify()


def test_verify_detects_inserted_record(chained):
    led, path = chained
    lines = path.read_text().splitlines(keepends=True)
    forged = json.dumps({"config_hash": "f", "sharpe": 9.9, "prev_hash": "0" * 64}) + "\n"
    path.write_text(lines[0] + forged + lines[1] + lines[2])
    with pytest.raises(ValueError, match="hash chain"):
        led.verify()


def test_legacy_records_tolerated_and_chained_onto(tmp_path):
    # Pre-chain files (records without prev_hash) still verify, and new
    # records chain onto the last legacy line's bytes.
    path = tmp_path / "legacy.jsonl"
    legacy = [
        {"config_hash": "x", "sharpe": 0.1, "n_obs": 100},
        {"config_hash": "y", "sharpe": 0.2, "n_obs": 100},
    ]
    with path.open("w") as f:
        for rec in legacy:
            f.write(json.dumps(rec) + "\n")
    led = ExperimentLedger(path)
    assert led.verify() == 0  # nothing chained yet, nothing broken

    led.record({"a": 1}, 0.3, 100)
    lines = path.read_bytes().splitlines()
    new_rec = json.loads(lines[2])
    assert new_rec["prev_hash"] == hashlib.sha256(lines[1]).hexdigest()
    assert led.verify() == 1


def test_verify_rejects_unchained_record_after_chain_starts(chained):
    led, path = chained
    with path.open("a") as f:
        f.write(json.dumps({"config_hash": "z", "sharpe": 0.4}) + "\n")
    with pytest.raises(ValueError, match="missing prev_hash"):
        led.verify()


def test_committed_demo_ledger_verifies():
    # The committed legacy ledger must always pass verify().
    path = Path(__file__).resolve().parents[1] / "experiments" / "demo_ledger.jsonl"
    ExperimentLedger(path).verify()
