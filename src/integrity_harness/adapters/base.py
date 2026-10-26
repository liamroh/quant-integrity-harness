"""Engine-agnostic strategy evaluation: CPCV splits, net pricing, one trial.

`run_cpcv_evaluation` is the only sanctioned way to evaluate a Strategy:
it fits per purged split, prices positions net of costs on each contiguous
test block, assembles the phi(N, k) out-of-sample paths, and records ONE
ledger trial BEFORE the caller sees any number (constitution rule 1).

Return and cost conventions:
- Single asset. A position is a scalar in [-1, 1], a fraction of NAV.
- A position taken at bar t earns the return from bar t to bar t+1,
  credited at bar t+1. The first bar of each test block earns nothing
  (entered flat).
- Turnover is |delta position| / 2 per bar in round-trip units, matching
  `CostModel.round_trip_cost`, plus a forced close of any open position at
  the last bar of each test block. Each block is a self-contained round
  trip; costs are slightly overstated at block boundaries, never
  understated.
- The ledger records the mean per-path net Sharpe (provisional owner
  default); the full per-path Sharpe list is JSON in the record's notes.
- The config hash uses `params()` captured BEFORE the first fit, so
  train-time parameter selection does not destabilize the hash.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np
import pandas as pd

from ..costs import CostModel
from ..ledger import ExperimentLedger
from ..metrics import sharpe_ratio
from ..splitters import CombinatorialPurgedCV


@runtime_checkable
class Strategy(Protocol):
    """What the runner needs from a strategy.

    Contract:
    - `fit(data, train_idx)` must FULLY reset state: the runner reuses one
      object across splits (sklearn convention).
    - `positions(data, idx)` must be causal: the position at bar t may use
      only data with timestamp <= t. Purging handles label overlap; feature
      causality is the strategy's responsibility (constitution rule 3).
    - `params()` identifies the configuration for the ledger. Before the
      first fit it must describe the full search (e.g. the parameter grid),
      not a selected point.
    """

    name: str

    def params(self) -> dict[str, Any]: ...

    def fit(self, data: pd.DataFrame, train_idx: np.ndarray) -> None: ...

    def positions(self, data: pd.DataFrame, idx: np.ndarray) -> np.ndarray: ...


def spans_from_index(n_obs: int, horizon: int) -> pd.DataFrame:
    """Positional label spans for evenly spaced bars: t0 = i, t1 = i + horizon.

    Build spans by hand for irregular timestamps.
    """
    t0 = np.arange(n_obs, dtype=float)
    return pd.DataFrame({"t0": t0, "t1": t0 + float(horizon)})


@dataclass(frozen=True)
class CPCVEvaluationResult:
    config_hash: str
    path_net_returns: np.ndarray  # (n_paths, n_obs) per-period net returns
    path_sharpes: np.ndarray  # (n_paths,) per-period net Sharpes
    mean_sharpe: float
    n_obs: int


def _close_series(data: pd.DataFrame | pd.Series) -> np.ndarray:
    if isinstance(data, pd.Series):
        close = data.to_numpy(dtype=float)
    else:
        if "close" not in data.columns:
            raise ValueError("data must be a Series of closes or have a 'close' column")
        close = data["close"].to_numpy(dtype=float)
    if not np.isfinite(close).all() or (close <= 0).any():
        raise ValueError("close prices must be finite and positive")
    return close


def _price_block(
    close: np.ndarray,
    block_idx: np.ndarray,
    positions: np.ndarray,
    cost_model: CostModel,
    participation: float,
) -> np.ndarray:
    """Net per-period returns of holding `positions` over one contiguous block.

    Implements the module's return and cost conventions. Returns an array
    aligned with `block_idx`.
    """
    if len(block_idx) != len(positions):
        raise ValueError("positions must align with the block's bars")
    pos = np.asarray(positions, dtype=float)
    if not np.isfinite(pos).all() or (np.abs(pos) > 1.0 + 1e-9).any():
        raise ValueError(
            "position must be a finite NAV fraction in [-1, 1]; got "
            f"range [{pos.min():.4g}, {pos.max():.4g}]"
        )
    px = close[block_idx]
    gross = np.zeros(len(pos))
    gross[1:] = pos[:-1] * (px[1:] / px[:-1] - 1.0)
    # Round-trip units: enter from flat, per-bar position changes, forced
    # close of any open position at the last bar.
    turnover = np.empty(len(pos))
    turnover[0] = np.abs(pos[0]) / 2.0
    turnover[1:] = np.abs(np.diff(pos)) / 2.0
    turnover[-1] += np.abs(pos[-1]) / 2.0
    return gross - turnover * cost_model.round_trip_cost(participation)


def run_cpcv_evaluation(
    strategy: Strategy,
    data: pd.DataFrame | pd.Series,
    spans: pd.DataFrame,
    cv: CombinatorialPurgedCV,
    cost_model: CostModel,
    participation: float,
    ledger: ExperimentLedger,
    allow_rerun: bool = False,
) -> CPCVEvaluationResult:
    """Evaluate one strategy configuration through CPCV, net of costs.

    Records exactly one ledger trial (the mean per-path net Sharpe, with
    per-path Sharpes in notes) before returning any number to the caller.
    """
    close = _close_series(data)
    n_obs = len(close)
    if len(spans) != n_obs:
        raise ValueError(f"spans has {len(spans)} rows but data has {n_obs} bars")

    # Trial identity is fixed BEFORE any fitting.
    config = {
        "strategy": strategy.name,
        "params": strategy.params(),
        "cv": {
            "n_groups": cv.n_groups,
            "n_test_groups": cv.n_test_groups,
            "embargo_pct": cv.embargo_pct,
        },
        "costs": {
            "half_spread_bps": cost_model.half_spread_bps,
            "impact_coeff": cost_model.impact_coeff,
            "daily_vol": cost_model.daily_vol,
        },
        "participation": participation,
        "n_obs": n_obs,
    }

    groups = np.array_split(np.arange(n_obs), cv.n_groups)
    split_predictions: dict[tuple[int, ...], np.ndarray] = {}
    for split in cv.split(spans):
        strategy.fit(data, split.train_idx)
        net_full = np.full(n_obs, np.nan)
        for g in split.test_groups:
            blk = groups[g]
            pos = np.asarray(strategy.positions(data, blk), dtype=float)
            net_full[blk] = _price_block(close, blk, pos, cost_model, participation)
        split_predictions[split.test_groups] = net_full

    paths = cv.assemble_paths(n_obs, split_predictions)
    if np.isnan(paths).any():
        raise RuntimeError("path assembly left gaps; CPCV coverage is broken")
    path_sharpes = np.array([sharpe_ratio(p) for p in paths])
    mean_sharpe = float(np.mean(path_sharpes))

    # Record the trial before the caller sees any number.
    notes = json.dumps(
        {"path_sharpes": path_sharpes.tolist(), "n_paths": len(path_sharpes)}
    )
    h = ledger.record(
        config, mean_sharpe, n_obs=n_obs, notes=notes, allow_rerun=allow_rerun
    )
    return CPCVEvaluationResult(
        config_hash=h,
        path_net_returns=paths,
        path_sharpes=path_sharpes,
        mean_sharpe=mean_sharpe,
        n_obs=n_obs,
    )
