---
name: pr
description: Verify lint, tests, and ledger integrity, then push the current branch and open or update a pull request with a concise technical description. Use when the user runs /pr or asks to open, update, or ship a PR.
---

# /pr: verify, push, open a pull request

## 1. Preflight (all must pass; abort and report the failing output otherwise)

Run from the repo root:

```bash
ruff check .
python -m pytest -q
python scripts/verify_ledgers.py experiments
```

Also check the branch diff against main:

- If evaluation code (`metrics.py`, `pbo.py`, `ledger.py`, `splitters.py`) and strategy code change in the same branch, stop. CLAUDE.md rule 2 forbids bundling them; split the branch first.
- If any `experiments/*.jsonl` changed, confirm the change is append-only (`bash scripts/ledger_append_check.sh origin/main`).

Do not open the PR with failing checks. Never mask a failure with `|| true`, skips, or pipe tails that hide exit codes; check exit codes directly.

## 2. Push

```bash
git push -u origin <current-branch>
```

New branches are named `lr/<feature-description>`, short and kebab-case
(e.g. `lr/cpcv-path-reporting`). If work started on a differently named
branch, keep it only when a PR is already open on it; otherwise rename
before pushing.

## 3. Open or update the PR

Use the GitHub MCP tools. If a PR already exists for the branch, update its body instead of opening a second one.

## PR description requirements

Concise and pragmatic. Technical detail only; no flowery language, no marketing adjectives. Expand a point only when the reviewer needs it to evaluate the change.

Structure:

1. One-line summary of what the PR does.
2. Changes: short bullets, grouped by area. State behavior changes and API changes explicitly.
3. Rationale only where non-obvious. If evaluation code changed, include the written rationale required by CLAUDE.md rule 2.
4. Tests: what was run and the result (e.g. "39 passed, ruff clean"), plus any new test files and what they cover.
5. Known limitations or follow-ups, if any, as one short list.

Commits and PR text follow the repo writing style: plain language, no em or en dashes, authored as the repo owner with no AI attribution trailers (enforced by `.claude/settings.json`).
