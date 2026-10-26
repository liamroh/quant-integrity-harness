"""Tests for the engine-agnostic adapter core (Strategy protocol, CPCV runner).

Return and cost conventions under test (hand-computed numbers):
- A position taken at bar t earns the return from bar t to bar t+1, credited
  at bar t+1. The first bar of each test block earns nothing (entered flat).
- Turnover is |delta position| / 2 per bar in round-trip units, plus a forced
  close of any open position at the last bar of each block.
- One ledger trial is recorded per evaluation, before the caller sees any
  number; the recorded Sharpe is the mean per-path net Sharpe and the
  per-path Sharpes are JSON in the record's notes.
"""

import json

import numpy as np
import pandas as pd
import pytest

from integrity_harness.adapters.base import (
    CPCVEvaluationResult,
    Strategy,
    _price_block,
    run_cpcv_evaluation,
    spans_from_index,
)
from integrity_harness.costs import CostModel
from integrity_harness.ledger import ExperimentLedger, config_hash
from integrity_harness.metrics import sharpe_ratio
from integrity_harness.splitters import CombinatorialPurgedCV

# Zero impact so the round-trip cost is exactly 2 * half_spread = 2e-4.
CM = CostModel(half_spread_bps=1.0, impact_coeff=0.0, daily_vol=0.02)
RT = CM.round_trip_cost(0.0)


class ConstantStrategy:
    """Holds a fixed position on every test bar. fit() is a no-op reset."""

    name = "constant"

    def __init__(self, position=1.0):
        self.position = position
        self.fit_calls = 0

    def params(self):
        return {"position": self.position}

    def fit(self, data, train_idx):
        self.fit_calls += 1

    def positions(self, data, idx):
        return np.full(len(idx), self.position)


def test_strategy_protocol():
    assert isinstance(ConstantStrategy(), Strategy)


def test_spans_from_index_worked_example():
    spans = spans_from_index(5, horizon=3)
    np.testing.assert_array_equal(spans["t0"].to_numpy(), [0.0, 1, 2, 3, 4])
    np.testing.assert_array_equal(spans["t1"].to_numpy(), [3.0, 4, 5, 6, 7])


def test_price_block_worked_example():
    # Prices 100 -> 101 -> 99 -> 100.98, full long position throughout.
    # Bar returns: +1%, -101/99... precisely r1=0.01, r2=-2/101, r3=0.02.
    # gross = [0, 0.01, -2/101, 0.02]
    # turnover = [0.5 (enter), 0, 0, 0.5 (forced close)] round-trip units
    # net = gross - turnover * RT with RT = 2e-4.
    close = np.array([100.0, 101.0, 99.0, 100.98])
    pos = np.ones(4)
    net = _price_block(close, np.arange(4), pos, CM, participation=0.0)
    expected = np.array(
        [0.0 - 0.5 * RT, 0.01, -2.0 / 101.0, 0.02 - 0.5 * RT]
    )
    np.testing.assert_allclose(net, expected, rtol=0, atol=1e-15)


def test_price_block_position_changes_charge_turnover():
    # Positions [0.5, -0.5, -0.5]: enter 0.25 RT, flip 0.5 RT, close 0.25 RT.
    close = np.array([100.0, 100.0, 100.0])  # flat prices: gross = 0
    pos = np.array([0.5, -0.5, -0.5])
    net = _price_block(close, np.arange(3), pos, CM, participation=0.0)
    np.testing.assert_allclose(net, [-0.25 * RT, -0.5 * RT, -0.25 * RT])
    # Total cost equals total round-trip units traded times RT.
    assert net.sum() == pytest.approx(-1.0 * RT)


def _run(strategy, n_obs=60, close=None, ledger_path=None, tmp_path=None):
    if close is None:
        rng = np.random.default_rng(0)
        close = 100.0 * np.cumprod(1.0 + rng.normal(0.0003, 0.01, n_obs))
    data = pd.DataFrame({"close": close})
    spans = spans_from_index(n_obs, horizon=2)
    cv = CombinatorialPurgedCV(n_groups=6, n_test_groups=2, embargo_pct=0.01)
    ledger = ExperimentLedger(ledger_path or (tmp_path / "ledger.jsonl"))
    res = run_cpcv_evaluation(
        strategy, data, spans, cv, CM, participation=0.0, ledger=ledger
    )
    return res, ledger


def test_run_cpcv_evaluation_end_to_end(tmp_path):
    strat = ConstantStrategy(position=1.0)
    res, ledger = _run(strat, tmp_path=tmp_path)

    assert isinstance(res, CPCVEvaluationResult)
    assert res.path_net_returns.shape == (5, 60)  # phi(6, 2) = 5 paths
    assert not np.isnan(res.path_net_returns).any()
    assert strat.fit_calls == 15  # one fit per split, C(6, 2)

    # Per-path Sharpes recompute from the path returns.
    np.testing.assert_allclose(
        res.path_sharpes,
        [sharpe_ratio(p) for p in res.path_net_returns],
    )
    assert res.mean_sharpe == pytest.approx(float(np.mean(res.path_sharpes)))
    assert res.n_obs == 60

    # Exactly one ledger trial, recorded with the mean per-path Sharpe and
    # the per-path Sharpes in notes.
    recs = ledger._records()
    assert len(recs) == 1
    assert recs[0]["config_hash"] == res.config_hash
    assert recs[0]["sharpe"] == pytest.approx(res.mean_sharpe)
    notes = json.loads(recs[0]["notes"])
    np.testing.assert_allclose(notes["path_sharpes"], res.path_sharpes)


def test_flat_prices_cost_only_invariant(tmp_path):
    # Flat prices: gross is zero everywhere, so each path's total net return
    # is exactly minus (blocks per path) * (1 RT unit) * RT. Each path covers
    # all 6 groups, one block each: sum = -6 * RT per path.
    strat = ConstantStrategy(position=1.0)
    res, _ = _run(strat, close=np.full(60, 100.0), tmp_path=tmp_path)
    np.testing.assert_allclose(
        res.path_net_returns.sum(axis=1), np.full(5, -6.0 * RT), atol=1e-15
    )


def test_config_hash_uses_prefit_params(tmp_path):
    class GridStrategy(ConstantStrategy):
        name = "grid"

        def __init__(self):
            super().__init__(position=0.5)
            self.selected = None

        def params(self):
            if self.selected is None:
                return {"grid": [1, 2, 3]}
            return {"selected": self.selected}

        def fit(self, data, train_idx):
            super().fit(data, train_idx)
            self.selected = 2  # params() changes after the first fit

    strat = GridStrategy()
    res, ledger = _run(strat, tmp_path=tmp_path)
    rec = ledger._records()[0]
    # The hash reflects the PRE-fit params, so train-time selection does not
    # destabilize it, and it recomputes from the recorded config.
    assert rec["config"]["params"] == {"grid": [1, 2, 3]}
    assert res.config_hash == rec["config_hash"] == config_hash(rec["config"])


def test_positions_out_of_bounds_rejected(tmp_path):
    strat = ConstantStrategy(position=2.0)  # |pos| > 1 is not a NAV fraction
    with pytest.raises(ValueError, match="position"):
        _run(strat, tmp_path=tmp_path)


def test_duplicate_config_refused_without_rerun(tmp_path):
    path = tmp_path / "ledger.jsonl"
    _run(ConstantStrategy(1.0), ledger_path=path, tmp_path=tmp_path)
    with pytest.raises(ValueError, match="already in ledger"):
        _run(ConstantStrategy(1.0), ledger_path=path, tmp_path=tmp_path)
