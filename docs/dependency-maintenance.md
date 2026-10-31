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

References: [Renovate Python support](https://docs.renovatebot.com/modules/manager/pep621/),
[Renovate GitHub Actions support](https://docs.renovatebot.com/modules/manager/github-actions/),
and [uv upgrade behavior](https://docs.astral.sh/uv/concepts/projects/sync/#upgrading-locked-package-versions).
