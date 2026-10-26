import numpy as np
import pytest

from integrity_harness.metrics import (
    annualization_factor_lo,
    deflated_sharpe_ratio,
    expected_max_sharpe,
    min_track_record_length,
    probabilistic_sharpe_ratio,
    sharpe_ratio,
)


def test_expected_max_sharpe_worked_example():
    # Crash-course worked example: N=100, sqrt(V)=0.5 -> E[max] ~ 1.267.
    assert expected_max_sharpe(100, 0.25) == pytest.approx(1.267, abs=0.005)


def test_expected_max_sharpe_matches_monte_carlo():
    # "You understand DSR when you can reproduce the E[max SR] simulation."
    rng = np.random.default_rng(0)
    n_trials, sd = 50, 0.4
    maxima = [rng.normal(0.0, sd, n_trials).max() for _ in range(20_000)]
    mc = float(np.mean(maxima))
    formula = expected_max_sharpe(n_trials, sd**2)
    assert formula == pytest.approx(mc, rel=0.03)


def test_dsr_worked_examples():
    # SR=1.0 with N=100 trials of SD 0.5 is BELOW the luck benchmark 1.267.
    low = deflated_sharpe_ratio(1.0, 100, 0.25, 1250)
    assert low < 0.5
    # SR=1.8 clears it decisively (crash-course example -> DSR ~ 1.0).
    high = deflated_sharpe_ratio(1.8, 100, 0.25, 1250)
    assert high > 0.999


def test_psr_monotonic_and_bounded():
    a = probabilistic_sharpe_ratio(0.1, 0.0, 252)
    b = probabilistic_sharpe_ratio(0.2, 0.0, 252)
    assert 0.0 < a < b < 1.0
    # Negative skew / fat tails reduce PSR.
    c = probabilistic_sharpe_ratio(0.2, 0.0, 252, skew=-1.0, kurtosis=6.0)
    assert c < b


def test_min_trl_behaviour():
    assert min_track_record_length(0.1, 0.2) == float("inf")
    short = min_track_record_length(0.3, 0.0)
    long = min_track_record_length(0.15, 0.0)
    assert short < long  # bigger edge -> shorter record needed


def test_lo_annualization_iid_reduces_to_sqrt_q():
    rng = np.random.default_rng(1)
    r = rng.normal(0, 0.01, 100_000)  # ~iid
    eta = annualization_factor_lo(r, 12)
    assert eta == pytest.approx(np.sqrt(12), rel=0.02)


def test_lo_annualization_warns_on_zero_variance_series():
    # A zero-variance series has no defined autocorrelation; the sqrt(q)
    # fallback is a behavior change and must not pass silently. (The
    # non-positive-denominator fallback also warns, but with the biased
    # ACF estimator that quantity is a Fejer-kernel integral of the
    # periodogram and provably nonnegative, so it has no reachable test.)
    r = np.zeros(100)
    with pytest.warns(RuntimeWarning, match="sqrt"):
        eta = annualization_factor_lo(r, 252)
    assert eta == pytest.approx(np.sqrt(252))


def test_lo_annualization_positive_autocorr_below_sqrt_q():
    rng = np.random.default_rng(2)
    e = rng.normal(0, 0.01, 100_000)
    r = np.empty_like(e)
    r[0] = e[0]
    for t in range(1, len(e)):  # AR(1), rho=0.3
        r[t] = 0.3 * r[t - 1] + e[t]
    assert annualization_factor_lo(r, 12) < np.sqrt(12)


def test_sharpe_zero_vol():
    assert sharpe_ratio(np.zeros(10)) == 0.0
