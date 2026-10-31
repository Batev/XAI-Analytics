import numpy as np
import pandas as pd
import pytest
from scipy.sparse import issparse
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from util import commons as c
from util.model import Algorithm
from util.split import Split, SplitTypes


def test_encoded_frame_preserves_row_identity(model):
    observations = model.X_test.iloc[[3, 1, 3]].copy()
    observations.index = pd.Index(["case-c", "case-a", "case-c"], name="case")
    expected = pd.DataFrame(
        model.model[0].transform(observations),
        index=observations.index,
        columns=model.features_ohe,
    )

    result = c.calculate_X_ohe(model.model, observations, model.features_ohe)

    pd.testing.assert_frame_equal(result, expected)


@pytest.mark.parametrize("sparse_output", [False, True], ids=["dense", "sparse"])
def test_mixed_encoding_preserves_training_state(sparse_output):
    training = pd.DataFrame(
        {
            "value": [0.0, 1.0, np.nan, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0],
            "region": ["north", "south", "west"] * 4,
        }
    )
    preprocessor = c.get_column_transformer(["value"], ["region"])
    preprocessor.set_params(sparse_threshold=1.0 if sparse_output else 0.0)
    pipeline = Pipeline(
        [("preprocessor", preprocessor), ("model", LogisticRegression(random_state=42))]
    ).fit(training, [0, 1] * 6)
    observations = pd.DataFrame(
        {"value": [1000.0, np.nan], "region": ["unknown", "north"]},
        index=pd.Index([51, 19], name="row"),
    )
    names = preprocessor.get_feature_names_out().tolist()
    expected = preprocessor.transform(observations)
    assert issparse(expected) == sparse_output
    expected = expected.toarray() if sparse_output else expected
    numeric = preprocessor.named_transformers_["num"]
    encoder = preprocessor.named_transformers_["cat"].named_steps["onehot"]
    statistics = numeric.named_steps["imputer"].statistics_.copy()
    mean = numeric.named_steps["scale"].mean_.copy()
    scale = numeric.named_steps["scale"].scale_.copy()
    categories = [values.copy() for values in encoder.categories_]
    probabilities = pipeline.predict_proba(training)

    result = c.calculate_X_ohe(pipeline, observations, names)

    pd.testing.assert_frame_equal(
        result, pd.DataFrame(expected, index=observations.index, columns=names)
    )
    np.testing.assert_array_equal(numeric.named_steps["imputer"].statistics_, statistics)
    np.testing.assert_array_equal(numeric.named_steps["scale"].mean_, mean)
    np.testing.assert_array_equal(numeric.named_steps["scale"].scale_, scale)
    for actual, original in zip(encoder.categories_, categories, strict=True):
        np.testing.assert_array_equal(actual, original)
    np.testing.assert_allclose(pipeline.predict_proba(training), probabilities)


def test_categorical_only_encoding_keeps_dropped_binary_schema():
    training = pd.DataFrame({"flag": ["no", "yes"] * 6})
    preprocessor = c.get_column_transformer([], ["flag"])
    pipeline = Pipeline(
        [("preprocessor", preprocessor), ("model", LogisticRegression(random_state=42))]
    ).fit(training, [0, 1] * 6)
    observations = pd.DataFrame({"flag": ["no", "unknown"]}, index=[17, 5])
    names = preprocessor.get_feature_names_out().tolist()
    expected = preprocessor.transform(observations)
    encoder = preprocessor.named_transformers_["cat"].named_steps["onehot"]
    categories = encoder.categories_[0].copy()
    dropped = encoder.drop_idx_.copy()

    result = c.calculate_X_ohe(pipeline, observations, names)

    pd.testing.assert_frame_equal(
        result, pd.DataFrame(expected, index=observations.index, columns=names)
    )
    assert result.shape == (2, 1)
    np.testing.assert_array_equal(encoder.categories_[0], categories)
    np.testing.assert_array_equal(encoder.drop_idx_, dropped)


def test_encoding_requires_fitted_preprocessor(data):
    observations, _ = data
    pipeline = Pipeline(
        [
            ("preprocessor", c.get_column_transformer(list(observations), [])),
            ("model", LogisticRegression()),
        ]
    )

    with pytest.raises(NotFittedError):
        c.calculate_X_ohe(pipeline, observations, list(observations))


def test_fill_model_keeps_training_scaler_and_consistent_encoded_predictions(model):
    c.fill_model(
        model,
        algorithm=Algorithm.LOGISTIC_REGRESSION,
        split=Split(SplitTypes.NORMAL),
        search_options={"param_grid": {}, "cv": 2, "n_jobs": 1},
    )
    preprocessor = model.model[0]
    np.testing.assert_allclose(
        preprocessor.named_transformers_["num"].named_steps["scale"].mean_,
        model.X_train.mean().to_numpy(),
    )
    for observations, encoded in (
        (model.X_train, model.X_train_ohe),
        (model.X_test, model.X_test_ohe),
    ):
        pd.testing.assert_index_equal(encoded.index, observations.index)
        np.testing.assert_allclose(encoded.to_numpy(), preprocessor.transform(observations))
        np.testing.assert_allclose(
            model.model[1].predict_proba(encoded.to_numpy()),
            model.model.predict_proba(observations),
        )
