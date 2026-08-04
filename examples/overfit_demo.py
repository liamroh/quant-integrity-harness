"""Demo: a data-mined 'strategy' that naive evaluation loves and the harness kills.

We generate pure-noise daily returns, mine N=200 random signal configurations,
pick the best in-sample Sharpe (looks great!), then show:
  1. the ledger's DSR gate blocks it,
  2. PBO says the selection is ~coin-flip out-of-sample,
  3. costs make it worse still.

Run:  python examples/overfit_demo.py
"""

import numpy as np

from integrity_harness import (
    CostModel,
    ExperimentLedger,
    net_returns,
    probability_of_backtest_overfitting,
    sharpe_ratio,
)

rng = np.random.default_rng(42)
T, N = 1250, 200  # 5 years daily, 200 mined configurations

# Pure noise "market": no strategy here can have real skill.
trial_returns = rng.normal(0.0, 0.01, size=(T, N))

ledger = ExperimentLedger("experiments/demo_ledger.jsonl")
sharpes = []
for j in range(N):
    sr = sharpe_ratio(trial_returns[:, j])
    sharpes.append(sr)
    ledger.record({"demo": "noise-mining", "config_id": j}, sr, T, allow_rerun=True)

best = int(np.argmax(sharpes))
best_sr = sharpes[best]
ann = best_sr * np.sqrt(252)
print(f"Mined {N} configs. Best in-sample Sharpe: {best_sr:.3f}/day (~{ann:.2f} annualized)")
print("A naive backtest report would ship this.\n")

# 1) DSR gate
try:
    ledger.report_gate(best_sr, n_obs=T)
    print("DSR gate: PASSED (unexpected!)")
except RuntimeError as e:
    print(f"DSR gate: {e}\n")

# 2) PBO
res = probability_of_backtest_overfitting(trial_returns, n_blocks=8)
print(f"PBO (CSCV, {res.n_partitions} partitions): {res.pbo:.2f}")
print("~0.5 means the 'best' pick ranks below the OOS median half the time.\n")

# 3) Costs
cm = CostModel(half_spread_bps=1.0, impact_coeff=0.5, daily_vol=0.02)
daily_turnover = np.full(T, 0.25)
net = net_returns(trial_returns[:, best], daily_turnover, cm, participation=0.02)
print(f"Net-of-cost Sharpe of the 'winner': {sharpe_ratio(net):.3f}/day")
print("\nConclusion: the harness converts a publishable-looking result into an honest zero.")
