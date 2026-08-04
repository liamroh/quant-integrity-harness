"""Leakage-free cross-validation for financial labels with spans.

Each observation i has a label span [t0_i, t1_i]: the feature timestamp and
the time at which its label resolves (e.g., a 20-day forward return resolves
20 days later). Overlapping spans between train and test leak information.

- Purging drops any training obs whose span intersects a test obs's span.
- Embargo additionally drops training obs starting within `embargo_pct` of
  the sample length after the test block ends (serial correlation leaks).
- CPCV forms all C(N, k) group combinations as test sets, producing
  phi(N, k) = (k / N) * C(N, k) distinct backtest paths.

Reference: Lopez de Prado, Advances in Financial Machine Learning, ch. 7, 12.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import comb

import numpy as np
import pandas as pd


def _purge_and_embargo(
    spans: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    embargo_pct: float,
) -> np.ndarray:
    """Return train_idx with overlapping / embargoed observations removed."""
    t0 = spans["t0"].to_numpy()
    t1 = spans["t1"].to_numpy()
    test_start = t0[test_idx].min()
    test_end = t1[test_idx].max()

    # Purge: train span [t0_j, t1_j] must not intersect [test_start, test_end].
    keep = (t1[train_idx] < test_start) | (t0[train_idx] > test_end)
    train_idx = train_idx[keep]

    # Embargo: drop train obs starting within h after the test block ends.
    if embargo_pct > 0:
        horizon = spans["t1"].max() - spans["t0"].min()
        h = horizon * embargo_pct
        keep = ~(
            (spans["t0"].to_numpy()[train_idx] > test_end)
            & (spans["t0"].to_numpy()[train_idx] <= test_end + h)
        )
        train_idx = train_idx[keep]
    return train_idx


class PurgedKFold:
    """K-fold on contiguous time blocks with purging and embargo.

    Parameters
    ----------
    n_splits : number of contiguous folds.
    embargo_pct : fraction of total time span embargoed after each test block
        (a common choice is 0.01).
    """

    def __init__(self, n_splits: int = 5, embargo_pct: float = 0.01):
        if n_splits < 2:
            raise ValueError("n_splits must be >= 2")
        self.n_splits = n_splits
        self.embargo_pct = embargo_pct

    def split(self, spans: pd.DataFrame):
        """Yield (train_idx, test_idx). `spans` needs columns t0, t1 sorted by t0."""
        _validate_spans(spans)
        n = len(spans)
        indices = np.arange(n)
        for fold in np.array_split(indices, self.n_splits):
            test_idx = fold
            train_idx = np.setdiff1d(indices, test_idx)
            train_idx = _purge_and_embargo(spans, train_idx, test_idx, self.embargo_pct)
            yield train_idx, test_idx


@dataclass(frozen=True)
class CPCVSplit:
    train_idx: np.ndarray
    test_idx: np.ndarray
    test_groups: tuple[int, ...]


class CombinatorialPurgedCV:
    """CPCV: all C(N, k) choices of k test groups out of N ordered groups.

    n_paths() returns phi(N, k) = (k / N) * C(N, k) -- e.g., N=6, k=2 gives
    15 splits and 5 backtest paths. Use `assemble_paths` to map per-split
    test predictions back into phi full out-of-sample paths.
    """

    def __init__(self, n_groups: int = 6, n_test_groups: int = 2, embargo_pct: float = 0.01):
        if not 1 <= n_test_groups < n_groups:
            raise ValueError("need 1 <= n_test_groups < n_groups")
        self.n_groups = n_groups
        self.n_test_groups = n_test_groups
        self.embargo_pct = embargo_pct

    def n_splits(self) -> int:
        return comb(self.n_groups, self.n_test_groups)

    def n_paths(self) -> int:
        return self.n_test_groups * comb(self.n_groups, self.n_test_groups) // self.n_groups

    def split(self, spans: pd.DataFrame):
        """Yield CPCVSplit for every combination of test groups."""
        _validate_spans(spans)
        n = len(spans)
        indices = np.arange(n)
        groups = np.array_split(indices, self.n_groups)
        for test_combo in combinations(range(self.n_groups), self.n_test_groups):
            test_idx = np.concatenate([groups[g] for g in test_combo])
            train_idx = np.setdiff1d(indices, test_idx)
            # Purge/embargo against each contiguous test group separately.
            for g in test_combo:
                train_idx = _purge_and_embargo(spans, train_idx, groups[g], self.embargo_pct)
            yield CPCVSplit(train_idx, test_idx, test_combo)

    def assemble_paths(self, n_obs: int, split_predictions: dict[tuple[int, ...], np.ndarray]):
        """Assemble phi(N, k) out-of-sample paths from per-split predictions.

        `split_predictions` maps test_groups -> full-length array with
        predictions filled at that split's test indices (NaN elsewhere).
        Each group appears as test in C(N-1, k-1) splits; assign its j-th
        appearance (in split order) to path j. Returns array of shape
        (n_paths, n_obs).
        """
        n_paths = self.n_paths()
        paths = np.full((n_paths, n_obs), np.nan)
        appearance_count: dict[int, int] = {g: 0 for g in range(self.n_groups)}
        indices = np.arange(n_obs)
        groups = np.array_split(indices, self.n_groups)
        for test_combo in combinations(range(self.n_groups), self.n_test_groups):
            preds = split_predictions[test_combo]
            for g in test_combo:
                path_id = appearance_count[g]
                paths[path_id, groups[g]] = preds[groups[g]]
                appearance_count[g] += 1
        return paths


def _validate_spans(spans: pd.DataFrame) -> None:
    if not {"t0", "t1"}.issubset(spans.columns):
        raise ValueError("spans must have columns 't0' and 't1'")
    if (spans["t1"] < spans["t0"]).any():
        raise ValueError("each span must satisfy t1 >= t0")
    if not spans["t0"].is_monotonic_increasing:
        raise ValueError("spans must be sorted by t0 (time-ordered)")
