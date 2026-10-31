"""Offline checks for distinguishing audit findings from scanner failures."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location(
    "audit_dependencies", Path(__file__).parents[1] / ".github/scripts/audit_dependencies.py"
)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_clean_audit():
    status, summary = audit.summarize({"dependencies": [{"name": "example", "vulns": []}]}, 0)
    assert status["status"] == "clean"
    assert "No known vulnerabilities" in summary


def test_audit_reports_each_finding_and_fix():
    report = {
        "dependencies": [
            {
                "name": "example",
                "version": "1.0",
                "vulns": [
                    {"id": "GHSA-example", "fix_versions": ["1.1"]},
                    {"id": "PYSEC-example", "fix_versions": []},
                ],
            }
        ]
    }
    status, summary = audit.summarize(report, 1)
    assert status["status"] == "findings"
    assert status["findings"] == 2
    assert "example | 1.0 | GHSA-example | 1.1" in summary
    assert "No fix published" in summary


@pytest.mark.parametrize(
    "report,code",
    [({}, 1), ({"dependencies": []}, 1), ({"dependencies": []}, 2), ({"dependencies": [None]}, 0)],
)
def test_scanner_failure_is_not_a_clean_report(report, code):
    with pytest.raises(ValueError):
        audit.summarize(report, code)


def test_skipped_packages_are_incomplete():
    status, summary = audit.summarize(
        {"dependencies": [{"name": "private-package", "skip_reason": "not on index"}]}, 0
    )
    assert status["status"] == "incomplete"
    assert "not on index" in summary


@pytest.mark.parametrize("scanner_output", [None, "invalid JSON"])
def test_scanner_failure_writes_error_status_and_discards_stale_report(
    tmp_path, monkeypatch, scanner_output
):
    output = tmp_path / "output"
    output.mkdir()
    report = output / "audit.json"
    report.write_text('{"dependencies": []}', encoding="utf-8")
    summary = tmp_path / "summary.md"
    monkeypatch.setattr(
        "sys.argv",
        [
            "audit",
            "--requirements",
            "input.txt",
            "--output-dir",
            str(output),
            "--summary",
            str(summary),
        ],
    )

    def failed_scan(*args, **kwargs):
        assert not report.exists()
        if scanner_output is not None:
            report.write_text(scanner_output, encoding="utf-8")
        return SimpleNamespace(returncode=1, stdout="", stderr="Network unavailable")

    monkeypatch.setattr(audit.subprocess, "run", failed_scan)
    assert audit.main() == 2
    assert json.loads((output / "status.json").read_text())["status"] == "error"
    assert "network failure" in summary.read_text()
    assert "Network unavailable" in (output / "scanner.log").read_text()
