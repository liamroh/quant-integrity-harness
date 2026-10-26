"""Tests for Harvey-Liu haircut Sharpe (Bonferroni, Holm, BHY).

Worked examples are hand-computed. The Monte Carlo test checks the
family-wise error rate that Bonferroni and Holm are supposed to control.
"""

import numpy as np
import pytest
from scipy import stats

from integrity_harness.haircut import (
    adjusted_p_values,
    haircut_sharpe,
    sharpe_p_value,
)

# Hand-computed example vector, N=4:
# sorted p:      [0.005, 0.01, 0.03, 0.04]
# Bonferroni:    4*p, capped at 1
# Holm (step-down, cumulative max over sorted):
#   [4*0.005, 3*0.01, 2*0.03, 1*0.04] -> [0.02, 0.03, 0.06, 0.04]
#   cummax -> [0.02, 0.03, 0.06, 0.06]
# BHY (step-up with c(4) = 1 + 1/2 + 1/3 + 1/4 = 25/12):
#   raw_k = c*N/k * p_(k) = [25/12*4*0.005, 25/12*2*0.01, 25/12*4/3*0.03, 25/12*0.04]
#         = [1/24, 1/24, 1/12, 1/12]
#   reverse cummin -> [1/24, 1/24, 1/12, 1/12]
P_VEC = [0.01, 0.04, 0.03, 0.005]


def test_sharpe_p_value_worked_example():
    # sr = 0.1 per period, n_obs = 101: t = 0.1 * sqrt(100) = 1.0,
    # p = 2 * (1 - Phi(1)) = 0.3173105078...
    assert sharpe_p_value(0.1, 101) == pytest.approx(0.31731050786, abs=1e-9)
    # Symmetric in sign.
    assert sharpe_p_value(-0.1, 101) == pytest.approx(0.31731050786, abs=1e-9)


def test_bonferroni_worked_example():
    adj = adjusted_p_values(P_VEC, "bonferroni")
    np.testing.assert_allclose(adj, [0.04, 0.16, 0.12, 0.02])


def test_holm_worked_example():
    adj = adjusted_p_values(P_VEC, "holm")
    np.testing.assert_allclose(adj, [0.03, 0.06, 0.06, 0.02])


def test_bhy_worked_example():
    adj = adjusted_p_values(P_VEC, "bhy")
    np.testing.assert_allclose(adj, [1 / 24, 1 / 12, 1 / 12, 1 / 24])


def test_adjustments_are_monotone_and_capped():
    rng = np.random.default_rng(20)
    p = rng.uniform(0, 1, 50)
    for method in ("bonferroni", "holm", "bhy"):
        adj = adjusted_p_values(p, method)
        assert (adj >= p - 1e-12).all()
        assert (adj <= 1.0).all()


def test_haircut_identity_with_no_other_trials():
    # With no other trials the adjusted p equals the raw p, so the haircut
    # Sharpe equals the observed Sharpe for every method.
    for method in ("bonferroni", "holm", "bhy"):
        res = haircut_sharpe(0.15, 1250, [], method=method)
        assert res.haircut_sr == pytest.approx(0.15, abs=1e-12)
        assert res.haircut == pytest.approx(0.0, abs=1e-12)


def test_haircut_grows_with_trials():
    rng = np.random.default_rng(21)
    others_10 = [sharpe_p_value(s, 1250) for s in rng.normal(0, 0.03, 10)]
    others_200 = [sharpe_p_value(s, 1250) for s in rng.normal(0, 0.03, 200)]
    sr = 0.08
    h10 = haircut_sharpe(sr, 1250, others_10, method="holm")
    h200 = haircut_sharpe(sr, 1250, others_200, method="holm")
    assert h200.haircut_sr < h10.haircut_sr < sr
    assert 0.0 <= h10.haircut < h200.haircut <= 1.0


def test_haircut_complete_when_adjusted_p_hits_one():
    # A marginal Sharpe against many trials: adjusted p caps at 1, so the
    # haircut Sharpe is 0 (complete haircut).
    res = haircut_sharpe(0.02, 300, [0.5] * 400, method="bonferroni")
    assert res.p_adjusted == 1.0
    assert res.haircut_sr == 0.0
    assert res.haircut == 1.0


def test_haircut_requires_positive_sr():
    with pytest.raises(ValueError, match="positive"):
        haircut_sharpe(-0.1, 1250, [])


def test_holm_controls_fwer_monte_carlo():
    # Under the complete null (all trials zero-skill), the probability that
    # ANY Holm-adjusted p falls below alpha must be <= alpha. 2000 sims of
    # N=20 trials with T=252: reject rate should sit near but not above
    # alpha = 0.05 (binomial SE ~ 0.005).
    rng = np.random.default_rng(22)
    alpha, n_sims, n_trials, t_obs = 0.05, 2000, 20, 252
    z = np.sqrt(t_obs - 1)
    rejections = 0
    for _ in range(n_sims):
        r = rng.normal(0.0, 0.01, size=(t_obs, n_trials))
        srs = r.mean(axis=0) / r.std(axis=0, ddof=1)
        p = 2.0 * stats.norm.sf(np.abs(srs) * z)
        if adjusted_p_values(p, "holm").min() < alpha:
            rejections += 1
    assert rejections / n_sims < alpha + 0.02
