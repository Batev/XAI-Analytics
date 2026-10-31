# XAI-Analytics

XAI-Analytics is a tool that opens the black-box of machine learning. It makes model interpretability easy. 
**XAI-Analytics** offers a wide range of features such as data visualization, data preprocessing, ML-model training and global and local ML-model interpreting.
The interactive Jupyter Notebook offers an user-friendly interface that gives the user full control over the tool. 
For a quick demo of **XAI-Analytics** jump to the [demo](#demo) section. For examples please refer to [examples](#examples).

## Getting started

The supported environment is **Python 3.12 on Windows and Linux**. Install
[uv](https://docs.astral.sh/uv/getting-started/installation/), clone this repository,
and run from its root (PowerShell or a Unix shell):

```sh
uv python install 3.12
uv sync --locked --group notebook --group dev
uv run --no-sync jupyter notebook XAI-Analytics.ipynb
```

uv creates `.venv`; an existing `venv` is preserved. The lockfile is the exact
development/CI environment. Python 3.7–3.9 and Binder's old environment are no longer
supported. No dependency bypasses or edits to installed packages are required.

For pip users, create a Python 3.12 virtual environment and run
`python -m pip install -r requirements.txt`. This generated export includes the
notebook interface. Dependency declarations live in `pyproject.toml`; update and
export them with:

```sh
uv lock
uv sync --locked --group notebook --group dev
uv export --locked --no-dev --group notebook --no-emit-project --output-file requirements.txt
```

### Development and verification

```sh
uv lock --check
uv pip check
uv run --no-sync pytest -q
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
```

CI runs the offline test suite on Windows and Linux and checks Ruff on Linux.
Coverage and JUnit reports are generated without an initial coverage threshold.
The suite includes a notebook smoke fixture, real classifier/explainer calls, and
three strict expected failures for preprocessing refitting, predicted-class
selection, and categorical names containing underscores. Their reasons are
documented alongside the regression tests.
Passing tests do not establish that the existing scientific correctness issues are resolved.
Historical examples still contain expensive searches and network downloads.

### Supported features

Tabular classification with logistic regression, decision trees, random forests,
SVC and XGBoost; SHAP; ELI5 for its supported estimators; LIME; optimized LIME with
CSI/VSI; RBO; local/CSV datasets; and notebook widgets. Regression is not a validated
workflow. ELI5 support is estimator-dependent, and unsupported combinations may raise.

Tests can pass keyword-only `param_grid`, `cv`, `n_jobs`, and `n_iter` to
`get_pipeline`, or pass these through `train_model(search_options=...)`.
`explain_single_instance` accepts `num_samples`, `stability_samples`,
`stability_calls`, `optimization_samples`, and `optimization_iterations`;
`Model.init_shap(nsamples=...)` controls SHAP's sample budget. Notebook defaults
are preserved. Optimized LIME continues using the released lime-stability package
and its original metrics and optimization algorithm.

### Legacy integrations

Skater, the old PDPbox adapter, and `xai` exploratory plots/Census/demographic-balancing integrations
are unavailable in the modern environment. Their code is preserved; direct calls
raise an actionable error and unavailable options are hidden or disabled in widgets.
SHAP dependence plots remain available and describe SHAP contributions, not PDPs.
The Census example requires a separately supplied dataset until its loader is replaced.

Original dependency files are preserved under `docs/legacy-environment/` for
provenance only. Do not install them into the supported environment. Maintained PDP/ICE
and permutation importance replacements are planned separately, followed by correctness
fixes, a computation package, and a standalone UI.

## <a name="demo"></a>Demo

_v1.0_

[![Demo](https://img.youtube.com/vi/x-FImaqYme8/0.jpg)](https://www.youtube.com/watch?v=x-FImaqYme8)

##  <a name="examples"></a>Examples
Example notebooks can be found in [examples](https://github.com/Batev/XAI-Analytics/tree/master). Quick HTML overview of the example notebooks:
* [Adult Census Income](https://htmlpreview.github.io/?https://github.com/Batev/XAI-Analytics/blob/master/XAI-Example-Income.html)
* [Risk Factors for Cervical Cancer](https://htmlpreview.github.io/?https://github.com/Batev/XAI-Analytics/blob/master/XAI-Example-Cervical.html)

## Known Issues

* ~~Error when training a model on a dataset without any *categorical* columns (or only one categorical column that is the target)~~ -> Fixed by [b87a276](https://github.com/Batev/XAI-Analytics/commit/b87a2769cc3e817be45c39f95321a0bd0873d5b2).
* ~~Internal *PDPBox* error when plotting a PDP: https://github.com/SauceCat/PDPbox/issues/40~~ -> Workaround by [6143001](https://github.com/Batev/XAI-Analytics/commit/61430014c91ee6a69e26930fec6d9b20693fe321).
* *Skater* cannot be installed with *pip*: https://github.com/amueller/word_cloud/issues/134 and https://github.com/oracle/Skater/issues/308
* ~~Not all interpretation techniques work with *XGBoost*.~~ Only *ELI5* does not work with *XGBoost* -> until [this pull request](https://github.com/TeamHG-Memex/eli5/pull/407) is merged.

### TODO

* ~~Add XAI-Analytics to [A gallery of interesting Jupyter Notebooks](https://github.com/jupyter/jupyter/wiki/A-gallery-of-interesting-Jupyter-Notebooks) when ALPHA version is released.~~
* Add auto-generated adversarial examples to some dataset.
* ~~Fix *XGBoost* interpretations bug.~~
* Add option for removing features from a dataset in the preprocessing step.
* Add option for normalizing/removing unknown values from a dataset in the preprocessing step.
* Add custom exceptions and raise them where needed.
* Improve logging -> do not show info messages in the notebooks, use display instead. Add more debug messages.

### Non-Goals

* Support for semi-supervised, unsupervised and reinforcement learning.
* Model interpretation of regression algorithms.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for branch naming, Conventional Commits,
pull requests, validation, and releases.

## Creators

This project was created by [Evgeni Batev](https://www.linkedin.com/in/evgeni-batev-49124b40/) under the supervision of [Ao.univ.Prof. Dr. Andreas Rauber](http://www.ifs.tuwien.ac.at/~andi/) at the [Vienna University of Technology](https://www.tuwien.at/en/).

## License

[BSD 3-Clause License](LICENSE)
