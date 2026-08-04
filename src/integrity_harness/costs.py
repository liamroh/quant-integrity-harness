"""Transaction cost model: half-spread + square-root market impact.

One-way cost (as a return) of trading Q shares in a name with daily volume V
and daily volatility sigma:

    cost = half_spread + Y * sigma * sqrt(Q / V)

with the impact prefactor Y empirically ~0.5-1.0. Net returns subtract
turnover x round-trip cost each period. Calibrate Y and spreads to your own
fills; the defaults are large-cap-ish placeholders, not truths.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CostModel:
    half_spread_bps: float = 1.0
    impact_coeff: float = 0.5  # Y in the square-root law
    daily_vol: float = 0.02  # sigma (daily, as a return)

    def one_way_cost(self, participation: float) -> float:
        """One-way cost (return units) at `participation` = Q / ADV."""
        if participation < 0:
            raise ValueError("participation must be >= 0")
        impact = self.impact_coeff * self.daily_vol * np.sqrt(participation)
        return self.half_spread_bps / 1e4 + impact

    def round_trip_cost(self, participation: float) -> float:
        return 2.0 * self.one_way_cost(participation)


def net_returns(
    gross_returns: np.ndarray,
    turnover: np.ndarray,
    cost_model: CostModel,
    participation: float,
) -> np.ndarray:
    """gross -> net: r_net[t] = r_gross[t] - turnover[t] * round_trip_cost.

    `turnover[t]` is the fraction of the portfolio traded at period t
    (1.0 = full round trip of the whole book).
    """
    g = np.asarray(gross_returns, dtype=float)
    to = np.asarray(turnover, dtype=float)
    if g.shape != to.shape:
        raise ValueError("gross_returns and turnover must have the same shape")
    if (to < 0).any():
        raise ValueError("turnover must be non-negative")
    return g - to * cost_model.round_trip_cost(participation)
