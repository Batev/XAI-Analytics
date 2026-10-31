# Dependency auditing

The Dependency audit workflow reports known vulnerabilities on pull requests,
pushes to `master`, manual dispatch, and Mondays at 06:00 UTC. It is advisory:
findings, skipped packages, and scanner/network failures do not block merging.
Its job is not a required branch-protection check.

The audit uses Python 3.12 on Linux and exports runtime, notebook, and development
dependencies from `uv.lock` to temporary storage. All versions are pinned;
pip-audit does not resolve dependencies or apply fixes. Environment markers are
evaluated for Linux, so this workflow does not establish Windows/macOS coverage.

The job summary lists each affected package/version, advisory ID, and available
fixed versions. Artifacts include the input export, raw JSON report, scanner log,
and a status file. A missing/invalid report or scanner error is reported as an
error, never as a clean scan. Skipped packages make the result incomplete.

For each finding, verify the advisory against the locked version, assess the
affected use, and prepare a focused dependency/compatibility PR. Record blockers
and follow-up actions in the baseline below. Do not suppress advisories broadly
or run automatic fixes. Making this audit required needs a separate policy PR
after the baseline has been triaged.

## Legacy dependency inputs

The three unsupported inputs formerly in `docs/legacy-environment/` were copied
byte-for-byte to `.local-artifacts/legacy-environment-backup/`. SHA-256 hashes were
verified before removing the tracked files; `manifest.json` in that ignored
directory records the source paths and hashes. Git history also preserves them.
Other checkouts can retrieve the files from the parent of the removal commit;
the local backup is not distributed. Never install those inputs as the supported
environment.

On 2026-10-31, GitHub reported 178 open dependency alerts, all associated with the
root `requirements.txt`, not the legacy directory. Some refer to versions absent
from the current environment. This manifest evidence does not establish that the
legacy directory caused those alerts, and no alerts were dismissed. Use fresh
locked-environment audit results to distinguish active findings from historical
dependency-graph state.

## Initial active-environment baseline

The first local Windows/Python 3.12 scan on 2026-10-31 reported no known
vulnerabilities across 162 packages and no skipped packages in the active export.
This differs from the 178 open GitHub alerts against the root manifest; no alert
was dismissed on the strength of this scan. Recheck the GitHub dependency graph
after the maintenance PRs merge and investigate any still-open advisory against
its affected range and current locked version. No active finding currently needs
a suppression or an automatic dependency fix.

The first Linux result is recorded in this PR's workflow summary and artifacts.
Subsequent scheduled scans provide updated evidence; a clean scan only covers
advisories known to the scanner at that time.
