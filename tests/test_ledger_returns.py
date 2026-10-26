"""Tests for return-series persistence in the ledger and ledger.pbo().

record(returns=...) stores the series in a content-addressed sidecar file
(experiments/returns/<sha16>.npy) with a SHA-256 recorded in the ledger
line, so the hash chain covers the sidecar's integrity. ledger.pbo()
assembles the CSCV matrix from stored series; trials without stored
returns (or with a mismatched length) are excluded and counted.
"""

import hashlib

import numpy as np
import pytest

from integrity_harness.ledger import ExperimentLedger
from integrity_harness.metrics import sharpe_ratio
from integrity_harness.pbo import probability_of_backtest_overfitting


def _record_with_returns(led, rng, n=6, t=400):
    series = []
    for i in range(n):
        r = rng.normal(0.0, 0.01, t)
        led.record({"s": "noise", "seed": i}, sharpe_ratio(r), t, returns=r)
        series.append(r)
    return series


def test_record_stores_and_reloads_returns(tmp_path):
    led = ExperimentLedger(tmp_path / "ledger.jsonl")
    rng = np.random.default_rng(11)
    r = rng.normal(0.001, 0.01, 300)
    h = led.record({"s": "a"}, sharpe_ratio(r), 300, returns=r)

    rec = [x for x in led._records() if x["config_hash"] == h][0]
    sidecar = (tmp_path / rec["returns_path"]).resolve()
    assert sidecar.is_file()
    # Worked example: the recorded digest is the SHA-256 of the sidecar bytes.
    assert rec["returns_sha256"] == hashlib.sha256(sidecar.read_bytes()).hexdigest()
    np.testing.assert_array_equal(led.load_returns(h), r)


def test_record_rejects_returns_n_obs_mismatch(tmp_path):
    led = ExperimentLedger(tmp_path / "ledger.jsonl")
    r = np.zeros(100) + 0.001
    with pytest.raises(ValueError, match="n_obs"):
        led.record({"s": "a"}, 0.1, 999, returns=r)


def test_record_warns_on_sharpe_returns_mismatch(tmp_path):
    led = ExperimentLedger(tmp_path / "ledger.jsonl")
    rng = np.random.default_rng(12)
    r = rng.normal(0.0, 0.01, 200)
    with pytest.warns(RuntimeWarning, match="does not match"):
        led.record({"s": "a"}, 5.0, 200, returns=r)


def test_ledger_pbo_agrees_with_direct_cscv(tmp_path):
    led = ExperimentLedger(tmp_path / "ledger.jsonl")
    rng = np.random.default_rng(13)
    series = _record_with_returns(led, rng, n=8, t=400)
    M = np.column_stack(series)
    direct = probability_of_backtest_overfitting(M, n_blocks=8)
    via_ledger = led.pbo(n_blocks=8)
    assert via_ledger.result.pbo == direct.pbo
    assert via_ledger.result.n_partitions == direct.n_partitions == 70
    assert via_ledger.n_included == 8
    assert via_ledger.n_excluded == 0


def test_ledger_pbo_excludes_and_counts(tmp_path):
    led = ExperimentLedger(tmp_path / "ledger.jsonl")
    rng = np.random.default_rng(14)
    _record_with_returns(led, rng, n=4, t=400)
    # One trial without returns, one with a different length: both excluded.
    led.record({"s": "no-returns"}, 0.1, 400)
    short = rng.normal(0.0, 0.01, 100)
    led.record({"s": "short"}, sharpe_ratio(short), 100, returns=short)
    res = led.pbo(n_blocks=8)
    assert res.n_included == 4
    assert res.n_excluded == 2


def test_ledger_pbo_requires_two_stored_series(tmp_path):
    led = ExperimentLedger(tmp_path / "ledger.jsonl")
    rng = np.random.default_rng(15)
    r = rng.normal(0.0, 0.01, 200)
    led.record({"s": "only"}, sharpe_ratio(r), 200, returns=r)
    led.record({"s": "scalar"}, 0.1, 200)
    with pytest.raises(ValueError, match="fewer than 2"):
        led.pbo()


def test_tampered_sidecar_detected(tmp_path):
    led = ExperimentLedger(tmp_path / "ledger.jsonl")
    rng = np.random.default_rng(16)
    r = rng.normal(0.0, 0.01, 200)
    h = led.record({"s": "a"}, sharpe_ratio(r), 200, returns=r)
    rec = led._records()[0]
    sidecar = tmp_path / rec["returns_path"]
    data = bytearray(sidecar.read_bytes())
    data[-1] ^= 0xFF
    sidecar.write_bytes(bytes(data))
    with pytest.raises(ValueError, match="sha256|tamper"):
        led.load_returns(h)
