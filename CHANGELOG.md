# Changelog

## Unreleased — 3.0.0 development

### Breaking changes

- Require Python 3.12 and replace the historical installation workarounds with a
  locked uv environment and a generated pip requirements export.
- Isolate Skater, the old PDPbox adapter, and xai-dependent exploratory plots,
  Census loading, and demographic balancing. Their historical code is preserved;
  unavailable integrations raise explicit errors and their controls are disabled
  or hidden.

### Added

- Offline integration tests for classifiers, SHAP, ELI5, standard and optimized
  LIME, CSI/VSI, RBO, dataset parsing, and notebook controls.
- Windows/Linux continuous integration, Ruff checks, and test/coverage reports.
- A documented correctness backlog with strict expected-failure regressions.
- Contribution, Conventional Commit, branch, merge, and release policies.
- Curated research archives with provenance inventories, separate from supported
  application code.

### Changed

- Update scientific dependencies and adapt removed scikit-learn APIs and modern
  SHAP output shapes.
- Add optional training/search and explanation budgets for small real test runs.

### Known limitations

Scientific correctness fixes remain a separate milestone. See
[the correctness backlog](docs/correctness-backlog.md). This development series
is not a stable release; the existing v1/v2 tags remain historical releases.
