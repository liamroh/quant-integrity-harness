# Research Constitution

This repo is a research-integrity harness for financial backtests. Agents
working in this repo MUST follow these rules. They are not suggestions.

## Non-negotiable methodology rules
1. **Every strategy evaluation is a ledger trial.** Any run that produces a
   Sharpe ratio must be recorded via `ExperimentLedger.record()` BEFORE the
   result is inspected. Informal "quick checks" count as trials.
2. **Never modify `metrics.py`, `pbo.py`, or the ledger to make a result
   look better.** Changes to evaluation code require a written rationale in
   the PR description and must not be bundled with strategy changes.
3. **No lookahead.** Features at time t may only use data with timestamp
   <= t. Standardization/winsorization parameters are fit on training data
   only. Label spans (t0, t1) must be declared for every dataset.
4. **All cross-validation goes through `PurgedKFold` or
   `CombinatorialPurgedCV`.** Plain sklearn KFold on time-series labels is a
   bug, not a choice.
5. **All reported Sharpes are per-period in computation** and annualized
   only at the display layer, using `annualization_factor_lo` (never a bare
   sqrt(252) on autocorrelated series).
6. **Results are net of costs.** Use `CostModel` + `net_returns` with
   participation and turnover stated explicitly.
7. **Reporting is gated.** Final results go through
   `ExperimentLedger.report_gate()` (DSR > 0.95 against the full ledger).
   If blocked, the honest conclusion is "not distinguishable from luck",
   write that, do not search for a framing that passes.

## Engineering rules
- Tests before implementation for any new statistical function; every
  formula gets (a) a worked-example test with hand-computed numbers and
  (b) where feasible, a Monte-Carlo agreement test.
- `ruff check` and `pytest` must pass before any commit.
- The experiment ledger file (`experiments/ledger.jsonl`) is append-only.
  Never edit or delete lines. Treat deletion as data fraud.

## Commit and PR conventions
- Branches are named `lr/<feature-description>`, short and kebab-case
  (e.g. `lr/cpcv-path-reporting`, `lr/fix-ledger-ci`).
- Commits are authored solely as Liam Roh <liamrohx@gmail.com>. No
  Co-Authored-By trailers, no AI attribution, no session links
  (enforced via `.claude/settings.json` attribution config).
- PR descriptions carry no attribution footers or generated-by lines.
- PR descriptions are concise and pragmatic: technical detail only, no
  flowery language, expand a point only when a reviewer needs it.
  Structure: one-line summary, changes, non-obvious rationale (required
  for evaluation-code changes per rule 2 above), tests run and results,
  known limitations.
- Writing style everywhere (code comments, docs, commit messages, PR
  text): plain language, no em or en dashes.
- Use the `/pr` skill to open PRs: it gates on ruff, pytest, and ledger
  verification before pushing.

## Agent workflow guidance
- Propose-and-test loops are allowed ONLY if every iteration is recorded in
  the ledger. Unregistered parameter sweeps are forbidden.
- When uncertain whether something counts as a "trial": it does.
