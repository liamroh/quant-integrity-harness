# integrity-harness

A research-integrity layer for financial backtests. Not another backtesting
engine — a statistical enforcement layer designed to wrap one (vectorbt,
Nautilus Trader, or your own), making the most common forms of backtest
self-deception mechanically difficult.

**What it enforces:**

- **Leakage-free cross-validation** — `PurgedKFold` and
  `CombinatorialPurgedCV` (CPCV) with label-span purging and embargo, per
  López de Prado (*Advances in Financial Machine Learning*, ch. 7, 12).
  CPCV generates φ(N,k) = (k/N)·C(N,k) full out-of-sample backtest paths so
  you evaluate a *distribution* of outcomes, not one lucky path.
- **Multiple-testing-aware statistics** — Probabilistic Sharpe Ratio,
  Deflated Sharpe Ratio (with the False Strategy Theorem E[max SR]
  benchmark), Minimum Track Record Length, Lo (2002) autocorrelation-aware
  annualization, and Probability of Backtest Overfitting via CSCV.
- **An append-only experiment ledger** — every configuration you evaluate is
  a recorded trial (JSONL: config hash, git SHA, timestamp, Sharpe). The
  ledger's true trial count and cross-trial variance feed the DSR, and
  `report_gate()` takes the raw return series plus a recorded trial's hash,
  recomputes the Sharpe, skew, and kurtosis itself, and refuses to bless
  any result that doesn't clear DSR > 0.95 against your *full* trial
  history. CI enforces that ledger files are never edited or deleted.
- **Net-of-cost evaluation** — half-spread + square-root-law market impact
  with explicit participation and turnover accounting.

## The point, in one demo

```bash
python examples/overfit_demo.py
```

Mines 200 random configurations from pure noise, "finds" a ~1.5 annualized
Sharpe, then shows the DSR gate blocking it, PBO calling the selection a
coin flip, and costs finishing the job. If your research process can't kill
this strategy, it can't be trusted to find a real one.

## Install & test

```bash
pip install -e ".[dev]"
pytest tests/ -q     # 20 tests: worked examples + Monte-Carlo agreement
ruff check src/ tests/
```

Every statistical function is tested two ways where feasible: against
hand-computed worked examples, and against Monte-Carlo simulation (e.g., the
E[max SR] closed form is verified against 20,000 simulated trial-maxima).

## Quick start

```python
import numpy as np, pandas as pd
from integrity_harness import (
    CombinatorialPurgedCV, ExperimentLedger, sharpe_ratio,
)

# Label spans: a 5-day forward-return label resolves 5 days after t0.
spans = pd.DataFrame({"t0": np.arange(1000.0)})
spans["t1"] = spans["t0"] + 5

cv = CombinatorialPurgedCV(n_groups=6, n_test_groups=2, embargo_pct=0.01)
ledger = ExperimentLedger("experiments/ledger.jsonl")

for split in cv.split(spans):          # 15 splits -> 5 OOS paths
    ...  # fit on split.train_idx, predict on split.test_idx

sr = sharpe_ratio(my_oos_returns)      # per-period, always
h = ledger.record({"model": "gbrt", "depth": 3}, sr, n_obs=len(my_oos_returns))
ledger.report_gate(my_oos_returns, h)  # raises unless DSR > 0.95
```

## Design rules (see CLAUDE.md)

The repo carries a research constitution for humans and AI agents alike:
every evaluation is a ledger trial, evaluation code never changes alongside
strategy code, all CV is purged, all Sharpes are per-period internally, and
blocked results are reported as "not distinguishable from luck" — because
that is what they are.

## References

- Bailey & López de Prado (2012), *The Sharpe Ratio Efficient Frontier* (PSR, MinTRL)
- Bailey & López de Prado (2014), *The Deflated Sharpe Ratio*
- Bailey, Borwein, López de Prado & Zhu (2017), *The Probability of Backtest Overfitting*
- Lo (2002), *The Statistics of Sharpe Ratios*
- López de Prado (2018), *Advances in Financial Machine Learning*

## Roadmap

- [ ] Nautilus Trader / vectorbt adapter layer
- [ ] CPCV path-distribution reporting (per-path DSR, drawdown envelope)
- [ ] Harvey–Liu haircut Sharpe
- [ ] Point-in-time data loaders (survivorship-bias-free universes)

MIT License.
