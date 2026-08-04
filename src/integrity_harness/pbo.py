"""Probability of Backtest Overfitting via Combinatorially Symmetric CV.

Bailey, Borwein, Lopez de Prado & Zhu (2017): partition the T x N matrix of
per-period returns for N trials into S row-blocks; over all C(S, S/2)
in-sample/out-of-sample assignments, find the best in-sample trial and its
relative rank out-of-sample. PBO = fraction of assignments where the best
in-sample trial falls at or below the out-of-sample median (logit <= 0).
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np

from .metrics import sharpe_ratio


@dataclass(frozen=True)
class PBOResult:
    pbo: float
    logits: np.ndarray
    n_partitions: int


def probability_of_backtest_overfitting(
    trial_returns: np.ndarray, n_blocks: int = 8
) -> PBOResult:
    """Compute PBO from a (T, N) matrix of per-period returns for N trials."""
    M = np.asarray(trial_returns, dtype=float)
    if M.ndim != 2:
        raise ValueError("trial_returns must be 2-D (T periods x N trials)")
    if n_blocks % 2 != 0:
        raise ValueError("n_blocks must be even")
    T, n_trials = M.shape
    if n_trials < 2:
        raise ValueError("need at least 2 trials")
    blocks = np.array_split(np.arange(T), n_blocks)

    logits = []
    for is_combo in combinations(range(n_blocks), n_blocks // 2):
        is_rows = np.concatenate([blocks[b] for b in is_combo])
        oos_rows = np.setdiff1d(np.arange(T), is_rows)
        sr_is = np.array([sharpe_ratio(M[is_rows, j]) for j in range(n_trials)])
        sr_oos = np.array([sharpe_ratio(M[oos_rows, j]) for j in range(n_trials)])
        best = int(np.argmax(sr_is))
        # Relative rank of the best-IS trial among OOS Sharpes, in (0, 1).
        rank = (np.sum(sr_oos < sr_oos[best]) + 0.5 * np.sum(sr_oos == sr_oos[best])) / n_trials
        rank = np.clip(rank, 1e-9, 1 - 1e-9)
        logits.append(np.log(rank / (1.0 - rank)))

    logits_arr = np.array(logits)
    return PBOResult(
        pbo=float(np.mean(logits_arr <= 0)),
        logits=logits_arr,
        n_partitions=len(logits_arr),
    )
