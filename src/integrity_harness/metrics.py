"""Backtest statistics: Sharpe estimation, PSR, DSR, MinTRL, E[max SR].

All Sharpe ratios in this module are PER-PERIOD (non-annualized) unless a
function explicitly annualizes. Mixing annualized and per-period units is the
most common implementation bug in this literature.

References:
- Lo (2002), "The Statistics of Sharpe Ratios," FAJ 58(4).
- Bailey & Lopez de Prado (2012), "The Sharpe Ratio Efficient Frontier."
- Bailey & Lopez de Prado (2014), "The Deflated Sharpe Ratio."
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy import stats

EULER_MASCHERONI = 0.5772156649015329


def sharpe_ratio(returns: np.ndarray, rf: float = 0.0) -> float:
    """Per-period Sharpe ratio of a return series."""
    r = np.asarray(returns, dtype=float) - rf
    sd = r.std(ddof=1)
    if sd == 0:
        return 0.0
    return float(r.mean() / sd)


def annualization_factor_lo(returns: np.ndarray, q: int) -> float:
    """Lo (2002) annualization factor eta_q under autocorrelation.

    SR(q) = eta_q * SR, with
    eta_q = q / sqrt(q + 2 * sum_{k=1}^{q-1} (q - k) * rho_k).

    Under i.i.d. returns (all rho_k = 0) this reduces to sqrt(q). Positive
    autocorrelation gives eta_q < sqrt(q): naive sqrt-scaling OVERSTATES SR.
    """
    r = np.asarray(returns, dtype=float)
    r = r - r.mean()
    T = len(r)
    gamma0 = float(r @ r) / T
    if gamma0 == 0:
        warnings.warn(
            "annualization_factor_lo: zero-variance series has no defined "
            "autocorrelation; falling back to sqrt(q).",
            RuntimeWarning,
            stacklevel=2,
        )
        return float(np.sqrt(q))
    s = 0.0
    for k in range(1, q):
        if k >= T:
            break
        rho_k = (float(r[k:] @ r[:-k]) / T) / gamma0
        s += (q - k) * rho_k
    denom = q + 2.0 * s
    if denom <= 0:
        # Defensive: with the biased (1/T) ACF estimator this quantity is a
        # Fejer-kernel integral of the periodogram, hence nonnegative, and
        # finite-sample attenuation keeps it strictly positive in every case
        # we could construct. Warn loudly if an input ever reaches it.
        warnings.warn(
            f"annualization_factor_lo: q + 2*sum(rho) = {denom:.4f} <= 0 "
            "(extreme negative autocorrelation); falling back to sqrt(q). "
            "Inspect the series before trusting any annualized figure.",
            RuntimeWarning,
            stacklevel=2,
        )
        return float(np.sqrt(q))
    return float(q / np.sqrt(denom))


def probabilistic_sharpe_ratio(
    observed_sr: float,
    benchmark_sr: float,
    n_obs: int,
    skew: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """PSR: P(true SR > benchmark_sr | observed track record).

    PSR = Phi( (SR - SR*) * sqrt(T - 1)
               / sqrt(1 - skew*SR + ((kurt - 1)/4) * SR^2) )

    Both `observed_sr` and `benchmark_sr` must be per-period. `kurtosis` is
    raw (normal = 3), not excess.
    """
    var_term = 1.0 - skew * observed_sr + ((kurtosis - 1.0) / 4.0) * observed_sr**2
    if var_term <= 0:
        raise ValueError(
            "Non-positive variance term; check skew/kurtosis inputs "
            f"(got {var_term:.4f})."
        )
    z = (observed_sr - benchmark_sr) * np.sqrt(n_obs - 1) / np.sqrt(var_term)
    return float(stats.norm.cdf(z))


def expected_max_sharpe(n_trials: int, trial_sr_var: float) -> float:
    """False Strategy Theorem: E[max SR] over N zero-skill trials.

    E[max] ~= sqrt(V) * [(1 - gamma) * Phi^-1(1 - 1/N)
                         + gamma * Phi^-1(1 - 1/(N*e))]
    with gamma the Euler-Mascheroni constant. Assumes independent trials;
    correlated trials reduce the effective N, making this an upper-ish bound
    on the benchmark (i.e., conservative for acceptance).
    """
    if n_trials < 2:
        return 0.0
    g = EULER_MASCHERONI
    q1 = stats.norm.ppf(1.0 - 1.0 / n_trials)
    q2 = stats.norm.ppf(1.0 - 1.0 / (n_trials * np.e))
    return float(np.sqrt(trial_sr_var) * ((1.0 - g) * q1 + g * q2))


def deflated_sharpe_ratio(
    observed_sr: float,
    n_trials: int,
    trial_sr_var: float,
    n_obs: int,
    skew: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """DSR = PSR with the benchmark set to E[max SR] under the null.

    Accept a strategy only if DSR is high (e.g., > 0.95). `n_trials` must
    count EVERY configuration tried, including informal ones -- that is the
    entire point of the experiment ledger.
    """
    sr_star = expected_max_sharpe(n_trials, trial_sr_var)
    return probabilistic_sharpe_ratio(observed_sr, sr_star, n_obs, skew, kurtosis)


def min_track_record_length(
    observed_sr: float,
    benchmark_sr: float,
    confidence: float = 0.95,
    skew: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """MinTRL: observations needed to conclude true SR > SR* at `confidence`.

    MinTRL = 1 + [1 - skew*SR + ((kurt-1)/4)*SR^2] * (z / (SR - SR*))^2
    """
    if observed_sr <= benchmark_sr:
        return float("inf")
    z = stats.norm.ppf(confidence)
    var_term = 1.0 - skew * observed_sr + ((kurtosis - 1.0) / 4.0) * observed_sr**2
    return float(1.0 + var_term * (z / (observed_sr - benchmark_sr)) ** 2)
