#!/usr/bin/env bash
# Enforce the append-only contract for committed ledger files.
#
# Usage: ledger_append_check.sh BASE_COMMIT [HEAD_COMMIT]
#
# For every experiments/**.jsonl file that differs between BASE_COMMIT and
# HEAD_COMMIT (default HEAD), require that the base version is a byte prefix
# of the head version. Appending records is allowed; editing existing bytes,
# truncating, or deleting the file fails the check. A plain
# `git diff --diff-filter=MD` cannot express this contract because git
# classifies an append as a modification (M).
set -euo pipefail

BASE="${1:?usage: ledger_append_check.sh BASE_COMMIT [HEAD_COMMIT]}"
HEAD_REF="${2:-HEAD}"

# Pushes of new branches (and workflow_dispatch) have no usable before-SHA.
if [[ "$BASE" =~ ^0+$ ]] || ! git cat-file -e "$BASE^{commit}" 2>/dev/null; then
  echo "No usable base commit ('$BASE'); nothing to compare against, skipping."
  exit 0
fi

# Compare against the merge base so appends that landed on the base branch
# after the fork point are not misread as edits in the head version.
if MB=$(git merge-base "$BASE" "$HEAD_REF" 2>/dev/null); then
  BASE="$MB"
fi

fail=0
tmpdir=$(mktemp -d)
trap 'rm -rf "$tmpdir"' EXIT

while IFS=$'\t' read -r status path; do
  [ -n "${path:-}" ] || continue
  case "$path" in
    *.jsonl) ;;
    *) continue ;;
  esac
  case "$status" in
    D)
      echo "::error::Ledger file deleted: $path (ledger files are append-only)"
      fail=1
      continue
      ;;
  esac
  git cat-file blob "$BASE:$path" > "$tmpdir/old"
  git cat-file blob "$HEAD_REF:$path" > "$tmpdir/new"
  old_size=$(wc -c < "$tmpdir/old")
  new_size=$(wc -c < "$tmpdir/new")
  if [ "$new_size" -lt "$old_size" ]; then
    echo "::error::Ledger file shrank: $path ($old_size -> $new_size bytes)"
    fail=1
    continue
  fi
  if ! cmp -s -n "$old_size" "$tmpdir/old" "$tmpdir/new"; then
    echo "::error::Existing ledger bytes changed in $path (append-only: new records may only be added at the end)"
    fail=1
    continue
  fi
  echo "OK: $path (base version is a prefix of the new version)"
done < <(git diff --no-renames --name-status --diff-filter=MD "$BASE" "$HEAD_REF" -- experiments/)

if [ "$fail" -ne 0 ]; then
  echo "::error::Ledger append-only check failed. Existing records are never edited or deleted."
  exit 1
fi
echo "Ledger append-only check passed."
