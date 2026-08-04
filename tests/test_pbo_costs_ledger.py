import numpy as np
import pytest

from integrity_harness.costs import CostModel, net_returns
from integrity_harness.ledger import ExperimentLedger
from integrity_harness.pbo import probability_of_backtest_overfitting


def test_pbo_pure_noise_is_high():
    # All-noise trials: the best in-sample pick is luck -> PBO ~ 0.5 on
    # average. A single matrix has high variance (partitions reuse the same
    # data), so average across independent noise matrices.
    rng = np.random.default_rng(3)
    pbos = []
    for _ in range(8):
        M = rng.normal(0, 0.01, size=(500, 20))
        res = probability_of_backtest_overfitting(M, n_blocks=8)
        pbos.append(res.pbo)
    assert res.n_partitions == 70  # C(8, 4)
    assert 0.35 < np.mean(pbos) < 0.65


def test_pbo_true_skill_is_low():
    rng = np.random.default_rng(4)
    M = rng.normal(0, 0.01, size=(500, 20))
    M[:, 7] += 0.004  # one genuinely superior trial
    res = probability_of_backtest_overfitting(M, n_blocks=8)
    assert res.pbo < 0.1


def test_cost_model_worked_example():
    # Crash-course example: sigma=2%, Y=0.5, participation=2% -> impact 14.1bps;
    # + 1bp half-spread -> one-way ~15.1bps, round-trip ~30.3bps.
    cm = CostModel(half_spread_bps=1.0, impact_coeff=0.5, daily_vol=0.02)
    one_way = cm.one_way_cost(0.02)
    assert one_way == pytest.approx(0.001514, abs=2e-5)
    assert cm.round_trip_cost(0.02) == pytest.approx(2 * one_way)


def test_net_returns_drag():
    cm = CostModel(half_spread_bps=1.0, impact_coeff=0.5, daily_vol=0.02)
    gross = np.full(12, 0.01)
    turnover = np.full(12, 1.0)  # full monthly round trip
    net = net_returns(gross, turnover, cm, participation=0.02)
    assert (net < gross).all()
    annual_drag = (gross - net).sum()
    assert annual_drag == pytest.approx(12 * cm.round_trip_cost(0.02))


def test_ledger_records_and_gates(tmp_path):
    led = ExperimentLedger(tmp_path / "ledger.jsonl")
    rng = np.random.default_rng(5)
    # Log 50 zero-skill trials with cross-trial SD ~0.35.
    for i in range(50):
        led.record({"strategy": "noise", "seed": i}, rng.normal(0, 0.35), 1250)
    assert led.n_trials() == 50

    # Duplicate config refused without allow_rerun.
    with pytest.raises(ValueError):
        led.record({"strategy": "noise", "seed": 0}, 0.1, 1250)
    led.record({"strategy": "noise", "seed": 0}, 0.1, 1250, allow_rerun=True)
    assert led.n_trials() == 51

    # A Sharpe below the luck benchmark E[max SR] is blocked; with 51 trials
    # of SD ~0.30 the benchmark is ~0.70, so 0.5 must fail and 2.5 must pass.
    with pytest.raises(RuntimeError, match="REPORT BLOCKED"):
        led.report_gate(0.5, n_obs=1250)
    assert led.report_gate(2.5, n_obs=1250) > 0.95


def test_ledger_rejects_thin_history(tmp_path):
    led = ExperimentLedger(tmp_path / "ledger.jsonl")
    led.record({"a": 1}, 0.5, 252)
    with pytest.raises(ValueError, match="fewer than 2"):
        led.dsr(0.5, 252)
