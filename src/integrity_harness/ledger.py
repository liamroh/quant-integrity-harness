"""Append-only experiment ledger: the multiple-testing control surface.

Every configuration ever evaluated gets one JSONL record: config hash,
timestamp, git SHA, and the realized per-period Sharpe. The ledger's trial
count and cross-trial Sharpe variance feed the Deflated Sharpe Ratio, so a
"final" result can only be reported with the honest N behind it.

Design rules:
- Append-only: records are never edited or deleted (deletion = fraud here).
- Tamper-evident: each record hash-chains to the previous one via
  `prev_hash`; `verify()` walks the chain and CI runs it on every
  committed ledger file.
- `record()` refuses duplicate config hashes unless `allow_rerun=True`,
  and reruns still append (they never overwrite). Note that every rerun
  counts as one more trial in the DSR's N, which deflates in the
  conservative direction (harder to pass, never easier).
- `report_gate()` takes the raw return series plus a recorded config hash,
  recomputes the statistics itself, and raises unless the DSR computed
  from the FULL ledger clears the threshold -- wire this into a
  pre-report hook.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

from .metrics import deflated_sharpe_ratio, sharpe_ratio

# Sentinel prev_hash for the first record of a hash-chained ledger.
GENESIS_HASH = "0" * 64


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

    def _raw_lines(self) -> list[bytes]:
        """Non-empty record lines as raw bytes (newline stripped)."""
        with self.path.open("rb") as f:
            return [ln.rstrip(b"\r\n") for ln in f if ln.strip()]

    def _records(self) -> list[dict[str, Any]]:
        return [json.loads(ln) for ln in self._raw_lines()]

    def record(
        self,
        config: dict[str, Any],
        sharpe: float,
        n_obs: int,
        notes: str = "",
        allow_rerun: bool = False,
    ) -> str:
        """Append one trial. Returns the config hash.

        Each record carries `prev_hash`, the SHA-256 of the previous
        record's raw line bytes (GENESIS_HASH for the first record), making
        the file a hash chain that `verify()` can check for tampering.
        """
        h = config_hash(config)
        lines = self._raw_lines()
        if not allow_rerun and any(json.loads(ln)["config_hash"] == h for ln in lines):
            raise ValueError(
                f"config_hash {h} already in ledger; pass allow_rerun=True to "
                "append a rerun (reruns are appended, never overwritten)."
            )
        prev_hash = GENESIS_HASH if not lines else hashlib.sha256(lines[-1]).hexdigest()
        rec = {
            "config_hash": h,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "git_sha": _git_sha(),
            "config": config,
            "sharpe": float(sharpe),
            "n_obs": int(n_obs),
            "notes": notes,
            "prev_hash": prev_hash,
        }
        with self.path.open("a") as f:
            f.write(json.dumps(rec, default=str) + "\n")
        return h

    def verify(self) -> int:
        """Walk the hash chain; return the number of verified links.

        Raises ValueError if any chained record's `prev_hash` does not match
        the SHA-256 of the preceding line, or if an unchained record appears
        after chaining has started. Records written before hash-chaining
        existed (no `prev_hash` field) are tolerated as a legacy prefix;
        the first chained record locks onto the last legacy line's bytes.

        The chain detects edits, insertions, and deletions anywhere except
        pure truncation of the newest records; committed files get
        truncation protection from the CI append-only prefix check.
        """
        chained = False
        verified = 0
        prev_line: bytes | None = None
        for i, ln in enumerate(self._raw_lines()):
            rec = json.loads(ln)
            if "prev_hash" not in rec:
                if chained:
                    raise ValueError(
                        f"{self.path}: record {i} is missing prev_hash after "
                        "chained records began; the ledger has been altered."
                    )
                prev_line = ln
                continue
            expected = (
                GENESIS_HASH if prev_line is None else hashlib.sha256(prev_line).hexdigest()
            )
            if rec["prev_hash"] != expected:
                raise ValueError(
                    f"{self.path}: hash chain broken at record {i} "
                    f"(config_hash {rec.get('config_hash', '?')}): prev_hash "
                    f"{rec['prev_hash'][:12]}... does not match the preceding "
                    f"record ({expected[:12]}...). The ledger has been edited, "
                    "reordered, or had records removed."
                )
            chained = True
            verified += 1
            prev_line = ln
        return verified

    def n_trials(self) -> int:
        return len(self._records())

    def trial_sharpe_variance(self) -> float:
        """Cross-trial variance of recorded Sharpes (feeds E[max SR]).

        Caveat: SR estimator variance scales roughly with 1/n_obs, so
        mixing trials with very different `n_obs` (or different return
        frequencies) muddies what this variance means in the False
        Strategy Theorem benchmark. Keep one frequency and broadly
        comparable sample sizes per ledger file.
        """
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
        returns: np.ndarray,
        config_hash: str,
        threshold: float = 0.95,
        sr_tol: float = 1e-6,
    ) -> float:
        """Raise unless the DSR of a RECORDED trial clears `threshold`.

        The gate does not trust caller-typed numbers. It requires:
        1. `config_hash` exists in the ledger (the trial was recorded
           before being reported, per the constitution).
        2. The per-period Sharpe recomputed from `returns` matches a
           recorded Sharpe for that hash within `sr_tol`, and the series
           length matches the recorded `n_obs`.
        Skew and raw kurtosis are then computed from the series itself, so
        fat tails and negative skew increase the DSR haircut instead of
        silently defaulting to Gaussian. Wire this into any 'publish
        results' path.
        """
        r = np.asarray(returns, dtype=float)
        if r.ndim != 1:
            raise ValueError("returns must be a 1-D per-period return series")
        matches = [rec for rec in self._records() if rec["config_hash"] == config_hash]
        if not matches:
            raise ValueError(
                f"config_hash {config_hash} not found in ledger; every "
                "evaluation must go through record() before it is reported."
            )
        sr = sharpe_ratio(r)
        n_obs = len(r)
        same_n = [rec for rec in matches if rec["n_obs"] == n_obs]
        if not same_n:
            raise ValueError(
                f"n_obs mismatch for {config_hash}: series has {n_obs} "
                f"observations, ledger has {[rec['n_obs'] for rec in matches]}."
            )
        if not any(abs(rec["sharpe"] - sr) <= sr_tol for rec in same_n):
            recorded = [round(rec["sharpe"], 6) for rec in same_n]
            raise ValueError(
                f"return series does not match ledger record(s) for "
                f"{config_hash}: recomputed sharpe {sr:.6f}, recorded "
                f"{recorded}. The gate only blesses recorded results."
            )
        d = self.dsr(sr, n_obs, float(stats.skew(r)), float(stats.kurtosis(r, fisher=False)))
        self._check_threshold(d, threshold)
        return d

    def report_gate_scalar(
        self,
        observed_sr: float,
        n_obs: int,
        threshold: float = 0.95,
        skew: float = 0.0,
        kurtosis: float = 3.0,
    ) -> float:
        """Deprecated escape hatch: trusts caller-supplied numbers.

        Prefer `report_gate(returns, config_hash)`. This path cannot verify
        the numbers against the ledger, and its Gaussian moment defaults
        understate the haircut exactly when it matters (fat tails,
        negative skew).
        """
        warnings.warn(
            "report_gate_scalar trusts caller-supplied numbers; use "
            "report_gate(returns, config_hash) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        d = self.dsr(observed_sr, n_obs, skew, kurtosis)
        self._check_threshold(d, threshold)
        return d

    def _check_threshold(self, d: float, threshold: float) -> None:
        if d <= threshold:
            raise RuntimeError(
                f"REPORT BLOCKED: DSR={d:.4f} <= {threshold} given "
                f"N={self.n_trials()} ledger trials. This result is not "
                "distinguishable from selection luck."
            )
