# Repository instructions

These instructions apply to the entire repository. Follow the Git, contribution,
and release policy in [CONTRIBUTING.md](CONTRIBUTING.md).

## Git workflow and naming

- Use short-lived, task-specific branches and pull requests targeting `master`.
  Follow the branch naming convention in CONTRIBUTING.md, including for automated
  work. Examples: `chore/modernize-foundation`, `fix/preprocessing-refit`.
- Use Conventional Commits 1.0.0 for every new commit and PR title. Use a lowercase
  type, an optional scope, and an imperative description of the actual change.
- Branch names and complete commit messages (subject, body, and trailers) must
  describe the work without mentioning Codex, AI agents, assistants, or similar
  tool/provenance labels. Do not add generated-by notices or tool attribution
  trailers. Preserve legitimate human authorship and existing historical commits.
- Keep commits focused; separate mechanical formatting from behavioral changes.
  Inspect the working tree first and preserve unrelated work. Stage explicit
  files or hunks so research artifacts and local environments are not included
  accidentally.
- Validate relevant tests and the required CI checks before merging. Prefer squash
  merging with a Conventional Commit message. Never force-push the default branch
  or rewrite published release tags.

## Verification

Use the environment and commands documented in README.md:

```sh
uv sync --locked --group notebook --group dev
uv lock --check
uv pip check
uv run --no-sync pytest -q
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
```

For documentation-only changes, check links, wording, and `git diff --check`;
running the scientific integration suite is unnecessary.

## Releases

- Keep the version in `pyproject.toml` as the version source and regenerate the
  lockfile and exported requirements when changing project metadata as needed.
- Preserve the existing `v1.*` and `v2.*` release history. The modernization series
  targets `3.0.0.dev0`, then `3.0.0a1`, release candidates, and `3.0.0`; do not reset
  this established project to a `0.x` release series.
- Follow the release checklist in CONTRIBUTING.md. Report verification actually
  performed, including any platform or manual checks that remain unverified.
