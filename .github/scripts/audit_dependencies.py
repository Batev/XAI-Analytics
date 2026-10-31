"""Run pip-audit and report findings separately from incomplete/failed scans."""

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path


def summarize(report, returncode):
    dependencies = report.get("dependencies") if isinstance(report, dict) else None
    if not isinstance(dependencies, list):
        raise ValueError("Scanner did not produce a dependency report")
    findings = []
    skipped = []
    for dependency in dependencies:
        if not isinstance(dependency, dict):
            raise ValueError("Scanner produced an invalid dependency entry")
        if dependency.get("skip_reason"):
            skipped.append(f"{dependency['name']}: {dependency['skip_reason']}")
        for vulnerability in dependency.get("vulns", []):
            findings.append(
                (
                    dependency["name"],
                    dependency["version"],
                    vulnerability["id"],
                    ", ".join(vulnerability.get("fix_versions", [])) or "No fix published",
                )
            )
    if returncode not in (0, 1) or (returncode == 1 and not findings):
        raise ValueError(f"Scanner failed with exit code {returncode}")
    status = "findings" if findings else "clean"
    if skipped:
        status = "incomplete"
    lines = [
        "## Dependency audit",
        "",
        f"Status: **{status}**. Advisory only; this job does not block merging.",
        f"{platform.system()}/Python {sys.version_info.major}.{sys.version_info.minor} markers apply; "
        "other platforms are not covered by this scan.",
        "",
    ]
    if findings:
        lines.extend(
            ["| Package | Installed | Advisory | Fix versions |", "| --- | --- | --- | --- |"]
        )
        for row in findings:
            lines.append("| " + " | ".join(str(value).replace("|", "\\|") for value in row) + " |")
    else:
        lines.append("No known vulnerabilities were reported in the audited packages.")
    if skipped:
        lines.extend(
            ["", "Skipped packages (scan incomplete):", *[f"- {item}" for item in skipped]]
        )
    lines.extend(["", "See the JSON report and scanner log artifacts for details.", ""])
    return {"status": status, "findings": len(findings), "skipped": skipped}, "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requirements", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "audit.json"
    # Never classify a previous report as the result of a failed new invocation.
    report_path.unlink(missing_ok=True)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip_audit",
            "--requirement",
            str(args.requirements),
            "--no-deps",
            "--disable-pip",
            "--progress-spinner",
            "off",
            "--format",
            "json",
            "--output",
            str(report_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    (args.output_dir / "scanner.log").write_text(result.stdout + result.stderr, encoding="utf-8")
    try:
        status, summary = summarize(
            json.loads(report_path.read_text(encoding="utf-8")), result.returncode
        )
    except (OSError, ValueError, KeyError, TypeError) as error:
        status = {"status": "error", "error": str(error), "returncode": result.returncode}
        summary = (
            "## Dependency audit\n\n"
            "**Scanner or network failure: no complete audit result is available.**\n\n"
            "Advisory only; inspect the scanner log artifact and rerun the job.\n"
        )
    (args.output_dir / "status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    with args.summary.open("a", encoding="utf-8") as output:
        output.write(summary)
    print(summary)
    return {"clean": 0, "findings": 1, "incomplete": 2, "error": 2}[status["status"]]


if __name__ == "__main__":
    sys.exit(main())
