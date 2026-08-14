"""integrity-harness: research-integrity layer for financial backtests.

Purged/embargoed CV, CPCV path generation, PSR/DSR/MinTRL, PBO (CSCV),
square-root-law transaction costs, and an append-only experiment ledger
that gates result reporting on the Deflated Sharpe Ratio.
"""

from .costs import CostModel, net_returns
from .ledger import ExperimentLedger, LedgerPBOResult, config_hash
from .metrics import (
    annualization_factor_lo,
    deflated_sharpe_ratio,
    expected_max_sharpe,
    min_track_record_length,
    probabilistic_sharpe_ratio,
    sharpe_ratio,
)
from .pbo import PBOResult, probability_of_backtest_overfitting
from .splitters import CombinatorialPurgedCV, CPCVSplit, PurgedKFold

__version__ = "0.1.0"

__all__ = [
    "CPCVSplit",
    "CombinatorialPurgedCV",
    "CostModel",
    "ExperimentLedger",
    "LedgerPBOResult",
    "PBOResult",
    "PurgedKFold",
    "annualization_factor_lo",
    "config_hash",
    "deflated_sharpe_ratio",
    "expected_max_sharpe",
    "min_track_record_length",
    "net_returns",
    "probabilistic_sharpe_ratio",
    "probability_of_backtest_overfitting",
    "sharpe_ratio",
]
