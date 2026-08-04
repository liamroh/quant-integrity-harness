"""Append-only experiment ledger: the multiple-testing control surface.

Every configuration ever evaluated gets one JSONL record: config hash,
timestamp, git SHA, and the realized per-period Sharpe. The ledger's trial
count and cross-trial Sharpe variance feed the Deflated Sharpe Ratio, so a
"final" result can only be reported with the honest N behind it.

Design rules:
- Append-only: records are never edited or deleted (deletion = fraud here).
- `record()` refuses duplicate config hashes unless `allow_rerun=True`,
  and reruns still append (they never overwrite).
- `report_gate()` raises unless the DSR computed from the FULL ledger
  clears the threshold -- wire this into a pre-report hook.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np

from .metrics import deflated_sharpe_ratio


def config_hash(config: dict[str, Any]) -> str:
    canon = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(canon.encode()).hexdigest()[:16]


def _git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        return out.stdout.strip() or "no-git"
    except (OSError, subprocess.TimeoutExpired):
        return "no-git"


class ExperimentLedger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)

    def _records(self) -> list[dict[str, Any]]:
        records = []
        with self.path.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    def record(
        self,
        config: dict[str, Any],
        sharpe: float,
        n_obs: int,
        notes: str = "",
        allow_rerun: bool = False,
    ) -> str:
        """Append one trial. Returns the config hash."""
        h = config_hash(config)
        if not allow_rerun and any(r["config_hash"] == h for r in self._records()):
            raise ValueError(
                f"config_hash {h} already in ledger; pass allow_rerun=True to "
                "append a rerun (reruns are appended, never overwritten)."
            )
        rec = {
            "config_hash": h,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "git_sha": _git_sha(),
            "config": config,
            "sharpe": float(sharpe),
            "n_obs": int(n_obs),
            "notes": notes,
        }
        with self.path.open("a") as f:
            f.write(json.dumps(rec, default=str) + "\n")
        return h

    def n_trials(self) -> int:
        return len(self._records())

    def trial_sharpe_variance(self) -> float:
        srs = [r["sharpe"] for r in self._records()]
        if len(srs) < 2:
            return 0.0
        return float(np.var(srs, ddof=1))

    def dsr(
        self,
        observed_sr: float,
        n_obs: int,
        skew: float = 0.0,
        kurtosis: float = 3.0,
    ) -> float:
        """DSR of `observed_sr` deflated by THIS ledger's full trial history."""
        n = self.n_trials()
        if n < 2:
            raise ValueError("ledger has fewer than 2 trials; nothing to deflate against")
        return deflated_sharpe_ratio(
            observed_sr, n, self.trial_sharpe_variance(), n_obs, skew, kurtosis
        )

    def report_gate(
        self,
        observed_sr: float,
        n_obs: int,
        threshold: float = 0.95,
        skew: float = 0.0,
        kurtosis: float = 3.0,
    ) -> float:
        """Raise unless DSR > threshold. Wire into any 'publish results' path."""
        d = self.dsr(observed_sr, n_obs, skew, kurtosis)
        if d <= threshold:
            raise RuntimeError(
                f"REPORT BLOCKED: DSR={d:.4f} <= {threshold} given "
                f"N={self.n_trials()} ledger trials. This result is not "
                "distinguishable from selection luck."
            )
        return d
