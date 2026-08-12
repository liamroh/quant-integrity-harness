import numpy as np
import pytest
from scipy import stats

from integrity_harness.costs import CostModel, net_returns
from integrity_harness.ledger import ExperimentLedger
from integrity_harness.metrics import deflated_sharpe_ratio, sharpe_ratio
from integrity_harness.pbo import probability_of_backtest_overfitting


def _record_noise_trials(led, rng, n=50, t=1250):
    """Record n zero-skill trials with Sharpes computed from real series."""
    for i in range(n):
        r = rng.normal(0.0, 0.01, t)
        led.record({"strategy": "noise", "seed": i}, sharpe_ratio(r), t)


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
    T = 1250
    _record_noise_trials(led, rng, n=50, t=T)
    assert led.n_trials() == 50

    # Duplicate config refused without allow_rerun; reruns append.
    with pytest.raises(ValueError):
        led.record({"strategy": "noise", "seed": 0}, 0.1, T)
    led.record({"strategy": "noise", "seed": 0}, 0.1, T, allow_rerun=True)
    assert led.n_trials() == 51

    # The gate takes the return series itself, not caller-typed numbers.
    # A zero-skill series is blocked; a genuinely strong one passes.
    weak = rng.normal(0.0, 0.01, T)
    h_weak = led.record({"strategy": "weak"}, sharpe_ratio(weak), T)
    with pytest.raises(RuntimeError, match="REPORT BLOCKED"):
        led.report_gate(weak, h_weak)

    strong = rng.normal(0.002, 0.01, T)  # per-period SR ~0.2
    h_strong = led.record({"strategy": "strong"}, sharpe_ratio(strong), T)
    assert led.report_gate(strong, h_strong) > 0.95


def test_report_gate_requires_recorded_trial(tmp_path):
    led = ExperimentLedger(tmp_path / "ledger.jsonl")
    rng = np.random.default_rng(6)
    _record_noise_trials(led, rng, n=5)
    r = rng.normal(0.002, 0.01, 1250)
    with pytest.raises(ValueError, match="not found in ledger"):
        led.report_gate(r, "deadbeefdeadbeef")


def test_report_gate_rejects_mismatched_series(tmp_path):
    # A typed-in Sharpe that does not match the series is refused: the gate
    # cannot be used to bless numbers the ledger never saw.
    led = ExperimentLedger(tmp_path / "ledger.jsonl")
    rng = np.random.default_rng(7)
    _record_noise_trials(led, rng, n=5)
    h = led.record({"strategy": "fabricated"}, 2.5, 1250)
    r = rng.normal(0.0, 0.01, 1250)
    with pytest.raises(ValueError, match="does not match"):
        led.report_gate(r, h)


def test_report_gate_rejects_mismatched_n_obs(tmp_path):
    led = ExperimentLedger(tmp_path / "ledger.jsonl")
    rng = np.random.default_rng(8)
    _record_noise_trials(led, rng, n=5)
    r = rng.normal(0.002, 0.01, 1250)
    h = led.record({"strategy": "s"}, sharpe_ratio(r), 999)
    with pytest.raises(ValueError, match="n_obs"):
        led.report_gate(r, h)


def test_report_gate_computes_moments_from_series(tmp_path):
    # Agreement test: the gate's DSR must equal deflated_sharpe_ratio fed
    # with the skew and raw kurtosis of the series itself (not Gaussian
    # defaults). Use a fat-tailed series so the moments actually matter.
    led = ExperimentLedger(tmp_path / "ledger.jsonl")
    rng = np.random.default_rng(9)
    T = 1250
    _record_noise_trials(led, rng, n=20, t=T)
    r = rng.standard_t(df=4, size=T) * 0.01 + 0.003
    h = led.record({"strategy": "fat-tails"}, sharpe_ratio(r), T)
    d = led.report_gate(r, h)
    expected = deflated_sharpe_ratio(
        sharpe_ratio(r),
        led.n_trials(),
        led.trial_sharpe_variance(),
        T,
        float(stats.skew(r)),
        float(stats.kurtosis(r, fisher=False)),
    )
    assert d == pytest.approx(expected)


def test_report_gate_scalar_is_deprecated(tmp_path):
    led = ExperimentLedger(tmp_path / "ledger.jsonl")
    rng = np.random.default_rng(10)
    _record_noise_trials(led, rng, n=10)
    with pytest.warns(DeprecationWarning, match="report_gate"):
        led.report_gate_scalar(2.5, 1250)


def test_ledger_rejects_thin_history(tmp_path):
    led = ExperimentLedger(tmp_path / "ledger.jsonl")
    led.record({"a": 1}, 0.5, 252)
    with pytest.raises(ValueError, match="fewer than 2"):
        led.dsr(0.5, 252)
