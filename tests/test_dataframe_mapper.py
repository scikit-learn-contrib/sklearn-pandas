import pytest

from pandas import DataFrame
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import Imputer, StandardScaler
import numpy as np

from sklearn_pandas import (
    DataFrameMapper,
    PassthroughTransformer,
    cross_val_score,
)


@pytest.fixture
def iris_dataframe():
    iris = load_iris()
    return DataFrame(
        data={
            iris.feature_names[0]: iris.data[:, 0],
            iris.feature_names[1]: iris.data[:, 1],
            iris.feature_names[2]: iris.data[:, 2],
            iris.feature_names[3]: iris.data[:, 3],
            "species": np.array([iris.target_names[e] for e in iris.target])
        }
    )


@pytest.fixture
def cars_dataframe():
    return pd.read_csv("tests/test_data/cars.csv.gz", compression='gzip')


def test_with_iris_dataframe(iris_dataframe):
    pipeline = Pipeline([
        ("preprocess", DataFrameMapper([
            ("petal length (cm)", PassthroughTransformer()),
            ("petal width (cm)", PassthroughTransformer()),
            ("sepal length (cm)", PassthroughTransformer()),
            ("sepal width (cm)", PassthroughTransformer()),
        ])),
        ("classify", SVC(kernel='linear'))
    ])
    data = iris_dataframe.drop("species", axis=1)
    labels = iris_dataframe["species"]
    scores = cross_val_score(pipeline, data, labels)
    assert scores.mean() > 0.96
    assert (scores.std() * 2) < 0.04


def test_with_car_dataframe(cars_dataframe):
    pipeline = Pipeline([
        ("preprocess", DataFrameMapper([
            ("description", CountVectorizer()),
        ])),
        ("classify", SVC(kernel='linear'))
    ])
    data = cars_dataframe.drop("model", axis=1)
    labels = cars_dataframe["model"]
    scores = cross_val_score(pipeline, data, labels)
    assert scores.mean() > 0.30


def test_list_transformers():
    dataframe = pd.DataFrame({"a": [1, np.nan, 3], "b": [1, 5, 7]})

    mapper = DataFrameMapper([
        (["a"], [Imputer(), StandardScaler()]),
        (["b"], StandardScaler()),
    ])
    dmatrix = mapper.fit_transform(dataframe)

    assert pd.isnull(dmatrix).sum() == 0  # no null values

    # all features have mean 0 and std deviation 1 (standardized)
    assert (abs(dmatrix.mean(axis=0) - 0) <= 1e-6).all()
    assert (abs(dmatrix.std(axis=0) - 1) <= 1e-6).all()
