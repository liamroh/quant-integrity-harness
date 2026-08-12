"""Tests for scripts/ledger_append_check.sh against a scratch git repo.

The contract: appending to a committed ledger file passes; editing existing
bytes, truncating, or deleting the file fails; non-ledger changes pass.
"""

import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "ledger_append_check.sh"

pytestmark = pytest.mark.skipif(
    shutil.which("git") is None or shutil.which("bash") is None,
    reason="requires git and bash",
)


def _git(cwd, *args):
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
    ).stdout.strip()


@pytest.fixture()
def repo(tmp_path):
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.name", "Test")
    _git(tmp_path, "config", "user.email", "test@example.com")
    ledger = tmp_path / "experiments" / "ledger.jsonl"
    ledger.parent.mkdir()
    ledger.write_text('{"config_hash": "a", "sharpe": 0.1}\n{"config_hash": "b", "sharpe": 0.2}\n')
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "base")
    base = _git(tmp_path, "rev-parse", "HEAD")
    return tmp_path, ledger, base


def _check(cwd, base):
    return subprocess.run(
        ["bash", str(SCRIPT), base], cwd=cwd, capture_output=True, text=True
    )


def _commit_all(cwd, msg="change"):
    _git(cwd, "add", "-A")
    _git(cwd, "commit", "-q", "-m", msg)


def test_append_passes(repo):
    cwd, ledger, base = repo
    with ledger.open("a") as f:
        f.write('{"config_hash": "c", "sharpe": 0.3}\n')
    _commit_all(cwd)
    result = _check(cwd, base)
    assert result.returncode == 0, result.stdout + result.stderr


def test_edit_fails(repo):
    cwd, ledger, base = repo
    text = ledger.read_text().replace('"sharpe": 0.1', '"sharpe": 9.9')
    ledger.write_text(text)
    _commit_all(cwd)
    result = _check(cwd, base)
    assert result.returncode == 1
    assert "changed" in result.stdout


def test_truncate_fails(repo):
    cwd, ledger, base = repo
    lines = ledger.read_text().splitlines(keepends=True)
    ledger.write_text("".join(lines[:-1]))
    _commit_all(cwd)
    result = _check(cwd, base)
    assert result.returncode == 1
    assert "shrank" in result.stdout


def test_delete_fails(repo):
    cwd, ledger, base = repo
    ledger.unlink()
    _commit_all(cwd)
    result = _check(cwd, base)
    assert result.returncode == 1
    assert "deleted" in result.stdout


def test_non_ledger_change_passes(repo):
    cwd, _, base = repo
    (cwd / "notes.txt").write_text("hello\n")
    _commit_all(cwd)
    result = _check(cwd, base)
    assert result.returncode == 0


def test_missing_base_skips(repo):
    cwd, _, _ = repo
    result = _check(cwd, "0" * 40)
    assert result.returncode == 0
    assert "skipping" in result.stdout
