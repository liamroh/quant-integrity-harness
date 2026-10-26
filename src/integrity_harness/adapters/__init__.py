"""Engine adapters: run external strategy engines through the harness."""

from .base import (
    CPCVEvaluationResult,
    Strategy,
    run_cpcv_evaluation,
    spans_from_index,
)

__all__ = [
    "CPCVEvaluationResult",
    "Strategy",
    "run_cpcv_evaluation",
    "spans_from_index",
]
