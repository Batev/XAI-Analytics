# Contributing

## Branches and pull requests

We use GitHub flow: `master` is the default integration branch. Create a short-lived
branch for each focused change and open a pull request against `master`. External
contributors can work from a fork. Draft pull requests are welcome for work in
progress.

Use `<type>/<short-description>` for work branches. Use lowercase words separated
by hyphens, with an optional issue number in the description:

```text
chore/modernize-foundation
fix/42-preprocessing-refit
feat/reference-selection
test/unseen-categories
ci/platform-checks
docs/installation-guide
```

Choose a type from the commit types below. Branch names and full commit messages,
including trailers, describe the change itself. Omit assistant/tool names and
automated-authoring labels, including Codex and AI-agent references. Do not add
generated-by notices or tool attribution trailers. Retain legitimate human
authorship.

Before editing, inspect the working tree and preserve unrelated changes. Stage
specific files or hunks. Keep commits focused and separate mechanical formatting
from behavior changes. Avoid committing downloaded dependencies, environments,
generated test reports, or uncurated research artifacts.

In a PR, explain the problem, resulting behavior, relevant validation, and any
compatibility or migration impact. Reference related issues, using `Closes #123`
when appropriate. Add a regression test alongside a bug fix.

## Conventional Commits

Every new commit and PR title follows
[Conventional Commits 1.0.0](https://www.conventionalcommits.org/en/v1.0.0/):

```text
<type>[optional scope][!]: <description>

[optional body]

[optional footers]
```

Use lowercase types and scopes, an imperative description, and no trailing full
stop in the subject. Aim for a subject of 72 characters or fewer. Use the body to
explain why the change is needed and any important tradeoffs.

| Type | Purpose |
| --- | --- |
| `feat` | Add a capability |
| `fix` | Correct a bug |
| `docs` | Change documentation |
| `test` | Add or improve tests |
| `ci` | Change continuous integration |
| `build` | Change dependencies, packaging, or build tooling |
| `refactor` | Restructure code without changing behavior |
| `perf` | Improve performance |
| `style` | Apply mechanical formatting |
| `chore` | Perform other maintenance |
| `revert` | Revert an earlier change |

Examples:

```text
fix(preprocessing): preserve fitted transformations
test(lime): cover optimized explanations
ci: validate Windows, Linux, and macOS environments
build!: require Python 3.12
```

For breaking changes, use `!` before the colon and include a `BREAKING CHANGE:`
footer describing the impact and migration. The footer is a project requirement
to help users migrate, even though the upstream specification permits `!` alone.

## Validation and merging

Follow README.md for installation and local verification. Required checks are
Windows tests, Linux tests, macOS tests, and Ruff lint/format checks.
The PR title check also enforces the Conventional Commit subject format and reruns
when a title is edited. CI cancels superseded PR runs and limits job duration;
pushes to `master` keep their own runs.

The required `lint` job also runs Actionlint on all GitHub Actions workflows and
the pinned Renovate configuration validator in strict repository mode. It rejects
invalid workflows, invalid Renovate settings, and settings needing migration.
The official validator containers are pinned by version and immutable digest;
Renovate maintains their references through its GitHub Actions manager. To run
the same checks locally, use the container references and arguments in
`.github/workflows/ci.yml` with Docker, mounting the repository as the working
directory.

Test/coverage reports and dependency-audit artifacts are retained for 14 days
after upload. This applies to new uploads; existing artifacts keep their original
expiration dates.

Tests include documented
strict expected failures for known scientific defects; a green build does not
mean those defects have been fixed.

The repository policy is to require PRs, passing checks, and resolved review
conversations on `master`, and to block force pushes and branch deletion. These
protections must also be configured in GitHub repository settings. Require zero
external approvals while there is only one active maintainer, and one independent
approval when a second maintainer regularly reviews contributions.

Use squash merge by default. Check that the final squash message follows
Conventional Commits and accurately describes the final change. Delete the work
branch after merging. Rebase personal branches when necessary; coordinate before
rewriting a shared branch and use `--force-with-lease` for an intentional update.
Preserve default-branch and published release history.

## Optional local checks

Install the locked development environment using the README commands, then opt
in to Git hooks with:

```sh
uv run --no-sync pre-commit install
```

To run checks without installing a hook:

```sh
uv run --no-sync pre-commit run --files util/model.py README.md
```

Local Ruff hooks use the same locked version, file coverage, and check-only
behavior as CI. Whitespace/final-newline hooks cover source, configuration, and
Markdown, preserving Markdown hard breaks; they may repair selected files.
Notebooks, HTML exports, datasets, and generated requirements are excluded.
Review any repairs before committing. Keep broad mechanical cleanup in its own
commit. The full scientific suite remains in CI and the documented verification
commands; it is not run on every commit. Hooks are optional, and can be removed
with `uv run --no-sync pre-commit uninstall`.

## Versioning and releases

See [Dependency maintenance](docs/dependency-maintenance.md) for the hosted update
service, generated pip export, and intentional package upgrades.

Use semantic versioning with Python-compatible development and prerelease forms:

- Patch: compatible bug fixes, such as `3.0.1`.
- Minor: compatible capabilities, such as `3.1.0`.
- Major: breaking changes, such as `4.0.0`.
- Development/prerelease: `3.0.0.dev0`, `3.0.0a1`, `3.0.0rc1`.

The existing history includes tags through `v2.1`. The modernization is the next
major series, targeting `3.0.0`. Use `3.0.0.dev0` during development and publish an
alpha before claiming stable compatibility and scientific correctness. Version
changes happen through focused PRs; ordinary feature PRs need not bump the version.

Release checklist:

1. Prepare a release PR updating the version in `pyproject.toml`, the lockfile and
   generated requirements where necessary, and `CHANGELOG.md`. Include migration
   notes, dependency/platform changes, and known limitations.
2. Require Windows, Linux, and macOS tests and Ruff checks to pass for the release commit.
   Record any applicable manual notebook verification and unresolved limitations.
3. Merge the release PR and create an annotated `v<version>` tag on that exact
   commit, for example `v3.0.0a1`.
4. Publish GitHub Release notes with installation instructions, changes, migration
   guidance, and known limitations. Mark alpha/RC releases as prereleases.
5. Preserve published tags and artifacts; publish a new version for corrections.

Create a maintenance branch such as `release/3.x` only when an older release needs
support alongside newer development. Backport selected fixes through PRs and test
them on that branch. When package publishing is introduced, use PyPI Trusted
Publishing and verify the built distributions before publishing.
