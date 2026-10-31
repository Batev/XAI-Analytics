# Correctness backlog

The modern environment establishes compatibility, not scientific validation.
Historical notebooks and saved results remain research artifacts.

| ID | Defect / required follow-up | Regression test |
| --- | --- | --- |
| COR-01 | Explanation transforms refit fitted preprocessing. Use training-fitted transform only. | test_regression_transform_does_not_refit |
| COR-02 | Predicted-class helper uses probability argmax instead of estimator prediction. | test_regression_prediction_class |
| COR-03 | Encoded names are split on underscores; dropped categories are lost. | test_regression_category_names |

These tests use strict xfail restricted to AssertionError. Unexpected passes require
removing the marker; unrelated exceptions fail normally.

Next, add failing regression tests and fixes for: resampling before CV; holdout-based
split selection; demographic balancing of evaluation populations; test-derived
SHAP/LIME references; inconsistent explained classes and contribution text; stale
explanation caches; signed versus magnitude ranking and RBO validation/ties;
SHAP dependence plots mislabeled as PDP; download certificate/status validation,
timeouts, size limits and duplicate ZIP requests; task inference and scoring.

Compatibility changes remove invalid modern estimator settings, but the full default
search grids remain expensive. Use explicit small search and explanation budgets in tests.
Regression and multiclass explanation support are not validated in this milestone.
