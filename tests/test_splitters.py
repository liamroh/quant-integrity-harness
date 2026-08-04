import numpy as np
import pandas as pd
import pytest

from integrity_harness.splitters import CombinatorialPurgedCV, PurgedKFold


def make_spans(n=120, label_horizon=5):
    t0 = np.arange(n, dtype=float)
    return pd.DataFrame({"t0": t0, "t1": t0 + label_horizon})


def spans_overlap(spans, i, j):
    return not (spans.t1[i] < spans.t0[j] or spans.t0[i] > spans.t1[j])


def test_purged_kfold_no_train_test_overlap():
    spans = make_spans()
    cv = PurgedKFold(n_splits=5, embargo_pct=0.0)
    for train_idx, test_idx in cv.split(spans):
        test_start = spans.t0.to_numpy()[test_idx].min()
        test_end = spans.t1.to_numpy()[test_idx].max()
        t0 = spans.t0.to_numpy()[train_idx]
        t1 = spans.t1.to_numpy()[train_idx]
        assert ((t1 < test_start) | (t0 > test_end)).all()


def test_embargo_removes_post_test_observations():
    spans = make_spans()
    lax = PurgedKFold(n_splits=5, embargo_pct=0.0)
    strict = PurgedKFold(n_splits=5, embargo_pct=0.05)
    sizes_lax = [len(tr) for tr, _ in lax.split(spans)]
    sizes_strict = [len(tr) for tr, _ in strict.split(spans)]
    # Embargo only bites for test blocks with training data after them.
    assert sum(sizes_strict) < sum(sizes_lax)


def test_cpcv_canonical_counts():
    # Lopez de Prado's canonical example: N=6, k=2 -> 15 splits, 5 paths.
    cv = CombinatorialPurgedCV(n_groups=6, n_test_groups=2)
    assert cv.n_splits() == 15
    assert cv.n_paths() == 5
    spans = make_spans()
    splits = list(cv.split(spans))
    assert len(splits) == 15


def test_cpcv_each_group_tested_equally():
    cv = CombinatorialPurgedCV(n_groups=6, n_test_groups=2, embargo_pct=0.0)
    spans = make_spans()
    counts = {g: 0 for g in range(6)}
    for s in cv.split(spans):
        for g in s.test_groups:
            counts[g] += 1
    # Each group is a test group in C(N-1, k-1) = C(5,1) = 5 splits.
    assert all(c == 5 for c in counts.values())


def test_cpcv_paths_fully_covered():
    n = 120
    cv = CombinatorialPurgedCV(n_groups=6, n_test_groups=2, embargo_pct=0.0)
    spans = make_spans(n)
    preds = {}
    for s in cv.split(spans):
        arr = np.full(n, np.nan)
        arr[s.test_idx] = 1.0
        preds[s.test_groups] = arr
    paths = cv.assemble_paths(n, preds)
    assert paths.shape == (5, n)
    assert not np.isnan(paths).any()  # every path is a complete OOS series


def test_rejects_unsorted_spans():
    spans = make_spans().iloc[::-1].reset_index(drop=True)
    with pytest.raises(ValueError):
        list(PurgedKFold().split(spans))
