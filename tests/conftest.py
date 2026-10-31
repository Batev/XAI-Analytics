import os
from pathlib import Path

os.environ["IPYTHONDIR"] = str(
    Path(__file__).resolve().parents[1] / ".local-artifacts" / "test-ipython"
)
os.environ["MPLBACKEND"] = "Agg"
os.environ["OMP_NUM_THREADS"] = "1"

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from util import commons as c
from util.model import Model, ModelType, ProblemType


@pytest.fixture(autouse=True)
def deterministic():
    np.random.seed(42)
    c._set_random_number(42)
    yield
    plt.close("all")


@pytest.fixture
def data():
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.normal(size=(60, 3)), columns=["a", "b", "c"])
    y = (X.a + X.b * 0.5 > 0).astype(int)
    return X, y


@pytest.fixture
def model(data):
    X, y = data
    pipeline = Pipeline(
        [
            ("preprocessor", c.get_column_transformer(list(X), [])),
            ("model", LogisticRegression(random_state=42)),
        ]
    ).fit(X, y)
    m = Model(0, "fixture", pipeline, X, y, ModelType(ProblemType.CLASSIFICATION))
    m.X_train, m.y_train = X, y
    m.X_test, m.y_test = X.iloc[:4].copy(), y.iloc[:4].copy()
    m.numerical_features, m.categorical_features = list(X), []
    m.features_ohe, m.idx2ohe = list(X), {}
    m.X_train_ohe = pd.DataFrame(pipeline[0].transform(X), columns=list(X))
    m.X_test_ohe = pd.DataFrame(pipeline[0].transform(m.X_test), columns=list(X))
    return m
