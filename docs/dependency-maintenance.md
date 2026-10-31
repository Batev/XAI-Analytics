# Dependency maintenance

`pyproject.toml` declares dependencies, `uv.lock` records the resolved environment,
and `requirements.txt` is the generated runtime/notebook export for pip users.
Python support changes and scientific algorithm changes need separate PRs.

## Hosted update service

Merge `renovate.json` into `master` before enabling the hosted Renovate GitHub App.
Grant the app access only to this repository. Installing it before the config is
merged may create an onboarding branch that does not follow our naming policy.

The configuration manages Python dependencies, GitHub Action commit references,
and the uv version supplied to setup-uv. Routine updates run on Mondays before
10:00 Europe/Sofia, with at most three open update branches/PRs. The hosted service
chooses its exact run time within that window. Major updates and weekly lockfile
refreshes require approval in the Dependency Dashboard. All PRs require manual
review and merging. Python support constraints and the exact LIME/lime-stability
pins are excluded from automated changes.

Generated requirements are not updated independently. For each Python update PR,
check out its branch, inspect the lockfile changes, then run:

```sh
uv sync --locked --group notebook --group dev
uv export --locked --no-dev --group notebook --no-emit-project --output-file requirements.txt
```

Commit the regenerated export to that same branch and push it. CI intentionally
blocks merging while the export is stale. Repeat after a bot rebase or refresh.
Do not give CI write access to repair the branch automatically.

Before accepting the first hosted update, verify its `build/deps-...` branch,
Conventional Commit title/message, lockfile changes, full Action SHA references,
manual merge policy, and the export consistency gate. Until the app is enabled,
this live verification remains outstanding.

## Intentional upgrades

Plain `uv lock` normally preserves versions that still satisfy the declarations.
To intentionally upgrade one package:

```sh
uv lock --upgrade-package pandas
uv sync --locked --group notebook --group dev
uv export --locked --no-dev --group notebook --no-emit-project --output-file requirements.txt
```

Run the README verification commands and require Windows, Linux, macOS, and Ruff
CI before merging. Keep each scientific dependency upgrade in its own PR and
include compatibility fixes and relevant regression coverage. Do not relax
dependency metadata to force an incompatible resolution.

The initial sequence is pandas, SciPy, then SHAP, selecting stable versions
compatible with Python 3.12 and the existing constraints. RBO currently requires
NumPy below 2; NumPy 2 and an RBO replacement need separate scientific validation.

### Upgrade assessment — 2026-10-31

Each package was resolved independently with `uv lock --upgrade-package <name>`.
All three resolutions left the lockfile unchanged: the locked versions remain
the newest stable versions compatible with the current dependency constraints.
The latest releases on PyPI are blocked as follows for Python 3.12:

| Package | Retained version | Latest assessed release | Required NumPy |
| --- | --- | --- | --- |
| pandas | 3.0.6 | [3.1.0](https://pypi.org/pypi/pandas/3.1.0/json) | `>=2.0.2` |
| SciPy | 1.17.1 | [1.18.1](https://pypi.org/pypi/scipy/1.18.1/json) | `>=2.0.0,<2.8` |
| SHAP | 0.49.1 | [0.54.1](https://pypi.org/pypi/shap/0.54.1/json) | `>=2` |

Explicit dry-run resolutions for each assessed release failed because
`rbo==0.1.3` requires `numpy>=1.18,<2.0`. Reproduce the checks without changing
the lockfile:

```sh
uv lock --dry-run --upgrade-package pandas==3.1.0
uv lock --dry-run --upgrade-package scipy==1.18.1
uv lock --dry-run --upgrade-package shap==0.54.1
```

These commands are expected to report unsatisfiable requirements. No dependency
metadata was overridden and no no-op upgrade PRs were created. The regenerated
pip export matches the unchanged runtime lock. The combined maintenance
environment passed 40 tests with the same two scientific expected failures,
plus Windows/Linux/macOS CI and Ruff.

Follow-up: separately evaluate an upstream RBO compatibility release or a
maintained replacement, validate RBO values and ranking semantics, and plan the
NumPy 2 migration. Then retry pandas, SciPy, and SHAP as individual upgrade PRs.
Do not combine that scientific migration with routine maintenance.

References: [Renovate Python support](https://docs.renovatebot.com/modules/manager/pep621/),
[Renovate GitHub Actions support](https://docs.renovatebot.com/modules/manager/github-actions/),
and [uv upgrade behavior](https://docs.astral.sh/uv/concepts/projects/sync/#upgrading-locked-package-versions).
