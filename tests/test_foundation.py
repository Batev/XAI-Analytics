import importlib
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from util import commons as c
from util import ui
from util.dataset import Dataset, Datasets
from util.model import Algorithm
from util.split import Split, SplitTypes


def test_imports():
    for name in ("commons", "dataset", "model", "split", "ui", "legacy"):
        importlib.import_module("util." + name)
    assert not {"skater", "pdpbox", "xai"} & sys.modules.keys()


@pytest.mark.parametrize("algorithm", list(Algorithm)[:5])
def test_classifier_fit(data, algorithm):
    X, y = data
    search = c.get_pipeline(
        c.get_column_transformer(list(X), []),
        algorithm,
        len(X),
        param_grid={},
        cv=2,
        n_jobs=1,
        n_iter=1,
    )
    search.fit(X, y)
    probabilities = search.predict_proba(X.iloc[:3])
    assert probabilities.shape == (3, 2)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=1e-6)
    assert search.predict(X).shape == (len(X),)


def test_mixed_preprocessing():
    X = pd.DataFrame({"n": [1.0, np.nan, 3.0, 4.0], "category": ["x", "y", "x", "y"]})
    preprocessor = c.get_column_transformer(["n"], ["category"])
    result = preprocessor.fit_transform(X)
    assert result.shape == (4, 2)
    assert np.isfinite(result).all()
    assert preprocessor.transform(pd.DataFrame({"n": [5.0], "category": ["z"]})).shape == (1, 2)


def test_training_entrypoint(data):
    from util.model import ModelType, ProblemType

    X, y = data
    mt = ModelType(ProblemType.CLASSIFICATION)
    mt.algorithm = Algorithm.DECISION_TREE
    fitted, train, yt, test, ytest = c.train_model(
        mt, Split(SplitTypes.NORMAL), X, y, search_options={"param_grid": {}, "cv": 2, "n_jobs": 1}
    )
    assert set(train.index).isdisjoint(test.index)
    assert len(fitted.predict(test)) == len(ytest)


def test_explainers(model):
    model.init_shap(nsamples=30)
    assert len(model.shap_values) == 2
    assert model.shap_values[0].shape == model.X_test_ohe.shape
    assert c.explain_single_instance_with_shap(model, 0) is not None
    assert c.plot_feature_importance_with_shap(model)
    html, weights = c.plot_feature_importance_with_eli5(model)
    assert html.data and weights
    explanation = c.explain_single_instance(c.LocalInterpreterType.LIME, model, 0, num_samples=100)
    assert explanation.as_list()


def test_optimized_lime(model):
    explanation = c.explain_single_instance(
        c.LocalInterpreterType.OPTIMIZED_LIME,
        model,
        0,
        num_samples=100,
        stability_samples=100,
        stability_calls=2,
        optimization_samples=2,
        optimization_iterations=2,
    )
    assert explanation.as_list()
    assert np.isfinite(model.optimized_kernel_width) and model.optimized_kernel_width > 0
    _, csi, vsi = c.explain_single_instance_with_lime_stability(
        model, 0, num_samples=100, stability_samples=100, stability_calls=2
    )
    assert all(np.isfinite(v) and 0 <= v <= 100 for v in (csi, vsi))


def test_rbo():
    assert c._calculate_rbo({"a": 2, "b": 1}, {"a": 2, "b": 1}, 2, 0.9) == 1
    assert c._calculate_rbo({"a": 2}, {"z": 2}, 1, 0.9) == 0


def test_widgets(model):
    grid = ui.generate_feature_importance_grid([model])
    assert grid.children[2].options == ("ELI5", "SHAP")
    assert ui.generate_pdp_grid([model]).children[2].options == ("SHAP",)
    local = ui.generate_local_interpretation_grid([model])
    assert "OPTIMIZED_LIME" in local.children[2].options


@pytest.mark.parametrize(
    "name",
    [
        "plot_feature_importance_with_skater",
        "plot_single_pdp_with_pdpbox",
        "plot_multi_pdp_with_pdpbox",
        "plot_single_pdp_with_skater",
        "plot_multi_pdp_with_skater",
    ],
)
def test_legacy_calls(name):
    import inspect

    fn = getattr(c, name)
    args = [None for p in inspect.signature(fn).parameters.values() if p.default is p.empty]
    with pytest.raises(NotImplementedError, match="README"):
        fn(*args)


def test_legacy_model(model):
    with pytest.raises(NotImplementedError, match="Skater"):
        model.init_skater()
    census = next(d for d in Datasets if d.value == 27)
    with pytest.raises(NotImplementedError, match="Census"):
        Dataset.get_dataframe(census)


def test_csv_http(monkeypatch):
    import httplib2

    monkeypatch.setattr(
        httplib2.Http, "request", lambda *a, **kw: ({"status": "200"}, b"x,y\n1,0\n2,1\n")
    )
    result = Dataset.from_url("fixture", "https://example.invalid/data.csv")
    assert result.df.to_dict("list") == {"x": [1, 2], "y": [0, 1]}
    monkeypatch.setattr(httplib2.Http, "request", lambda *a, **kw: ({"status": "404"}, b""))
    with pytest.raises(ConnectionError):
        Dataset.from_url("fixture", "https://example.invalid/data.csv")


def test_regression_transform_does_not_refit(model):
    before = model.model[0].named_transformers_["num"].named_steps["scale"].mean_.copy()
    probabilities = model.model.predict_proba(model.X_test)
    c.calculate_X_ohe(model.model, model.X_test + 100, model.features_ohe)
    np.testing.assert_allclose(
        model.model[0].named_transformers_["num"].named_steps["scale"].mean_, before
    )
    np.testing.assert_allclose(model.model.predict_proba(model.X_test), probabilities)


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="COR-02: class selection uses probability argmax instead of estimator.predict",
)
def test_regression_prediction_class():
    estimator = SimpleNamespace(
        predict_proba=lambda X: np.array([[0.6, 0.4]]), predict=lambda X: np.array([1])
    )
    result, _ = c._get_prediction_for_example(SimpleNamespace(model=estimator, X_test=None), 0)
    assert result == estimator.predict(None)[0]


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="COR-03: splitting encoded names on underscores corrupts feature and category names",
)
def test_regression_category_names():
    X = pd.DataFrame({"job_type": ["self_employed", "full_time"]})
    assert c.generate_idx2ohe_dict(
        X, ["job_type"], ["job_type_self_employed", "job_type_full_time"]
    ) == {0: ["self_employed", "full_time"]}


@pytest.mark.parametrize("name", ["imbalance_plot", "correlations"])
def test_legacy_eda(name):
    from util import legacy

    with pytest.raises(NotImplementedError, match="README"):
        getattr(legacy, name)(None)


def test_legacy_demographic_split(data):
    X, y = data
    with pytest.raises(NotImplementedError, match="demographic"):
        c._get_train_test_split(Split(SplitTypes.BALANCED, ["a"]), [], X, y)
