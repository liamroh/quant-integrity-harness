"""Harvey-Liu haircut Sharpe: multiple-testing-adjusted Sharpe ratios.

Harvey & Liu (2015), "Backtesting", Journal of Portfolio Management 42(1).
The observed Sharpe is converted to a two-sided p-value under the zero-skill
null, the p-value is adjusted for the number of trials (Bonferroni, Holm, or
Benjamini-Hochberg-Yekutieli), and the adjusted p-value is converted back to
a Sharpe. The gap between observed and haircut Sharpe is the multiple-testing
haircut.

All Sharpes here are PER-PERIOD, consistent with the rest of the package.
The t statistic uses sqrt(n_obs - 1), matching the PSR convention in
metrics.py (Harvey-Liu use sqrt(n_obs); the difference is negligible for
realistic sample sizes).

Relation to the DSR gate: DSR deflates against the expected maximum Sharpe
of the trial history; the haircut adjusts the trial's p-value directly.
They answer the same question from different angles. Reporting still goes
through ExperimentLedger.report_gate(); the haircut is a complementary
diagnostic.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

_METHODS = ("bonferroni", "holm", "bhy")


def sharpe_p_value(observed_sr: float, n_obs: int) -> float:
    """Two-sided p-value of a per-period Sharpe under the zero-skill null.

    t = SR * sqrt(n_obs - 1), p = 2 * (1 - Phi(|t|)).
    """
    t = observed_sr * np.sqrt(n_obs - 1)
    return float(2.0 * stats.norm.sf(abs(t)))


def adjusted_p_values(p_values: np.ndarray, method: str = "holm") -> np.ndarray:
    """Multiple-testing-adjusted p-values, returned in the input order.

    bonferroni: min(N * p, 1).
    holm: step-down; cumulative max of (N - k + 1) * p_(k) over sorted p.
    bhy: Benjamini-Hochberg-Yekutieli step-up with c(N) = sum_{i<=N} 1/i,
        valid under arbitrary dependence; controls FDR, not FWER.
    """
    if method not in _METHODS:
        raise ValueError(f"method must be one of {_METHODS}, got {method!r}")
    p = np.asarray(p_values, dtype=float)
    if p.ndim != 1 or len(p) == 0:
        raise ValueError("p_values must be a non-empty 1-D array")
    if (p < 0).any() or (p > 1).any():
        raise ValueError("p_values must lie in [0, 1]")
    n = len(p)
    if method == "bonferroni":
        return np.minimum(n * p, 1.0)
    order = np.argsort(p, kind="stable")
    p_sorted = p[order]
    if method == "holm":
        adj_sorted = np.minimum(np.maximum.accumulate((n - np.arange(n)) * p_sorted), 1.0)
    else:  # bhy
        c_n = float(np.sum(1.0 / np.arange(1, n + 1)))
        raw = np.minimum(c_n * n / np.arange(1, n + 1) * p_sorted, 1.0)
        adj_sorted = np.minimum.accumulate(raw[::-1])[::-1]
    adj = np.empty(n)
    adj[order] = adj_sorted
    return adj


@dataclass(frozen=True)
class HaircutResult:
    p_value: float
    p_adjusted: float
    haircut_sr: float
    haircut: float  # fraction of the observed Sharpe removed, in [0, 1]
    method: str
    n_tests: int


def haircut_sharpe(
    observed_sr: float,
    n_obs: int,
    other_p_values: np.ndarray,
    method: str = "holm",
) -> HaircutResult:
    """Haircut Sharpe of one trial against the p-values of the other trials.

    `other_p_values` are the p-values of every OTHER trial in the search
    (e.g. from the experiment ledger via sharpe_p_value on each record);
    the observed trial's own p-value is appended internally, so do not
    include it. With no other trials the haircut is zero by construction.

    The observed Sharpe must be positive: a haircut of a non-positive
    Sharpe is meaningless (there is nothing to deflate).
    """
    if observed_sr <= 0:
        raise ValueError("observed_sr must be positive to take a haircut")
    p = sharpe_p_value(observed_sr, n_obs)
    all_p = np.append(np.asarray(other_p_values, dtype=float), p)
    p_adj = float(adjusted_p_values(all_p, method)[-1])
    t_adj = float(stats.norm.isf(p_adj / 2.0))
    hc_sr = max(t_adj, 0.0) / float(np.sqrt(n_obs - 1))
    hc_sr = min(hc_sr, observed_sr)  # adjustment can only shrink
    return HaircutResult(
        p_value=p,
        p_adjusted=p_adj,
        haircut_sr=hc_sr,
        haircut=1.0 - hc_sr / observed_sr,
        method=method,
        n_tests=len(all_p),
    )
