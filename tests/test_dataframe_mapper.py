import pytest

# In py3, mock is included with the unittest standard library
# In py2, it's a separate package
try:
    from unittest.mock import Mock
except ImportError:
    from mock import Mock

from pandas import DataFrame
import pandas as pd
from scipy import sparse
from sklearn.cross_validation import cross_val_score
from sklearn.datasets import load_iris
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import Imputer, StandardScaler, OneHotEncoder
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from numpy.testing import assert_array_equal
import pickle

from sklearn_pandas.dataframe_mapper import DataFrameMapper
from sklearn_pandas.utils import handle_feature


class MockXTransformer(object):
    """
    Mock transformer that accepts no y argument.
    """
    def fit(self, X):
        return self

    def transform(self, X):
        return X


class MockTClassifier(object):
    """
    Mock transformer/classifier.
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X

    def predict(self, X):
        return True


class ToSparseTransformer(BaseEstimator, TransformerMixin):
    """
    Transforms numpy matrix to sparse format.
    """
    def fit(self, X):
        return self

    def transform(self, X):
        return sparse.csr_matrix(X)


@pytest.fixture
def simple_dataframe():
    return pd.DataFrame({'a': [1, 2, 3]})


@pytest.fixture
def complex_dataframe():
    return pd.DataFrame({'target': ['a', 'a', 'a', 'b', 'b', 'b'],
                         'feat1': [1, 2, 3, 4, 5, 6],
                         'feat2': [1, 2, 3, 2, 3, 4]})


def test_cols_string_array(simple_dataframe):
    """
    If an string specified as the columns, the transformer
    is called with a 1-d array as input.
    """
    df = simple_dataframe
    mock_transformer = Mock()
    mock_transformer.transform.return_value = np.array([1, 2, 3])  # do nothing
    mapper = DataFrameMapper([("a", mock_transformer)])

    mapper.fit_transform(df)
    args, kwargs = mock_transformer.fit.call_args
    assert args[0].shape == (3,)


def test_cols_list_column_vector(simple_dataframe):
    """
    If a one-element list is specified as the columns, the transformer
    is called with a column vector as input.
    """
    df = simple_dataframe
    mock_transformer = Mock()
    mock_transformer.transform.return_value = np.array([1, 2, 3])  # do nothing
    mapper = DataFrameMapper([(["a"], mock_transformer)])

    mapper.fit_transform(df)
    args, kwargs = mock_transformer.fit.call_args
    assert args[0].shape == (3, 1)


def test_handle_feature_2dim():
    """
    2-dimensional arrays are returned unchanged.
    """
    array = np.array([[1, 2], [3, 4]])
    assert_array_equal(handle_feature(array), array)


def test_handle_feature_1dim():
    """
    1-dimensional arrays are converted to 2-dimensional column vectors.
    """
    array = np.array([1, 2])
    assert_array_equal(handle_feature(array), np.array([[1], [2]]))


def test_selected_columns():
    """
    selected_columns returns a set of the columns appearing in the features
    of the mapper.
    """
    mapper = DataFrameMapper([
        ('a', None),
        (['a', 'b'], None)
    ])
    assert mapper._selected_columns == {'a', 'b'}


def test_unselected_columns():
    """
    selected_columns returns a list of the columns not appearing in the
    features of the mapper but present in the given dataframe.
    """
    df = pd.DataFrame({'a': [1], 'b': [2], 'c': [3]})
    mapper = DataFrameMapper([
        ('a', None),
        (['a', 'b'], None)
    ])
    assert 'c' in mapper._unselected_columns(df)


def test_default_false():
    """
    If default=False, non explicitly selected columns are discarded.
    """
    df = pd.DataFrame({'a': [1, 2, 3], 'b': [3, 5, 7]})
    mapper = DataFrameMapper([
        ('b', None)
    ], default=False)

    transformed = mapper.fit_transform(df)
    assert transformed.shape == (3, 1)


def test_default_none():
    """
    If default=None, non explicitly selected columns are passed through
    untransformed.
    """
    df = pd.DataFrame({'a': [1, 2, 3], 'b': [3, 5, 7]})
    mapper = DataFrameMapper([
        (['a'], OneHotEncoder())
    ], default=None)

    transformed = mapper.fit_transform(df)
    assert (transformed[:, 3] == np.array([3, 5, 7]).T).all()


def test_default_transformer():
    """
    If default=Transformer, non explicitly selected columns are applied this
    transformer.
    """
    df = pd.DataFrame({'a': [1, np.nan, 3], })
    mapper = DataFrameMapper([], default=Imputer())

    transformed = mapper.fit_transform(df)
    assert (transformed[:0] == np.array([1., 2., 3.])).all()


def test_list_transformers_single_arg(simple_dataframe):
    """
    Multiple transformers can be specified in a list even if some of them
    only accept one X argument instead of two (X, y).
    """
    mapper = DataFrameMapper([
        ('a', [MockXTransformer()])
    ])
    # doesn't fail
    mapper.fit_transform(simple_dataframe)


def test_list_transformers():
    """
    Specifying a list of transformers applies them sequentially to the
    selected column.
    """
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


def test_list_transformers_old_unpickle(simple_dataframe):
    mapper = DataFrameMapper([('a', [MockXTransformer()])])
    # simulate the mapper was created with < 1.0.0 code
    mapper.features = [('a', [MockXTransformer()])]
    mapper_pickled = pickle.dumps(mapper)

    loaded_mapper = pickle.loads(mapper_pickled)
    assert isinstance(loaded_mapper.pipeline, FeatureUnion)


def test_list_transformers_nofeatunion_unpickle(simple_dataframe):
    mapper = DataFrameMapper([('a', [MockXTransformer()])])
    # simulate the mapper was created with < 1.0.0 code
    del mapper.pipeline
    mapper_pickled = pickle.dumps(mapper)

    loaded_mapper = pickle.loads(mapper_pickled)
    assert isinstance(loaded_mapper.pipeline, FeatureUnion)


def test_default_old_unpickle(simple_dataframe):
    mapper = DataFrameMapper([('a', None)])
    # simulate the mapper was pickled before the ``default`` init argument
    # existed
    del mapper.default
    mapper_pickled = pickle.dumps(mapper)

    loaded_mapper = pickle.loads(mapper_pickled)
    loaded_mapper.fit_transform(simple_dataframe)  # doesn't fail


def test_sparse_features(simple_dataframe):
    """
    If any of the extracted features is sparse and "sparse" argument
    is true, the hstacked result is also sparse.
    """
    df = simple_dataframe
    mapper = DataFrameMapper([
        ("a", ToSparseTransformer())
    ], sparse=True)
    dmatrix = mapper.fit_transform(df)

    assert type(dmatrix) == sparse.csr.csr_matrix


def test_sparse_off(simple_dataframe):
    """
    If the resulting features are sparse but the "sparse" argument
    of the mapper is False, return a non-sparse matrix.
    """
    df = simple_dataframe
    mapper = DataFrameMapper([
        ("a", ToSparseTransformer())
    ], sparse=False)

    dmatrix = mapper.fit_transform(df)
    assert type(dmatrix) != sparse.csr.csr_matrix


def test_fit_with_optional_y_arg(complex_dataframe):
    """
    Transformers with an optional y argument in the fit method
    are handled correctly
    """
    df = complex_dataframe
    mapper = DataFrameMapper([(['feat1', 'feat2'], MockTClassifier())])
    # doesn't fail
    mapper.fit(df[['feat1', 'feat2']], df['target'])


def test_fit_with_required_y_arg(complex_dataframe):
    """
    Transformers with a required y argument in the fit method
    are handled and perform correctly
    """
    df = complex_dataframe
    mapper = DataFrameMapper([(['feat1', 'feat2'], SelectKBest(chi2, k=1))])

    # fit, doesn't fail
    ft_arr = mapper.fit(df[['feat1', 'feat2']], df['target'])

    # fit_transform
    ft_arr = mapper.fit_transform(df[['feat1', 'feat2']], df['target'])
    assert_array_equal(ft_arr, df[['feat1']].values)

    # transform
    t_arr = mapper.transform(df[['feat1', 'feat2']])
    assert_array_equal(t_arr, df[['feat1']].values)


# Integration tests with real dataframes

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
            ("petal length (cm)", None),
            ("petal width (cm)", None),
            ("sepal length (cm)", None),
            ("sepal width (cm)", None),
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


def test_get_params():
    pipeline = Pipeline([
        ("preprocess", DataFrameMapper([
            ("description", CountVectorizer()),
        ])),
        ("classify", SVC(kernel='linear'))
    ])
    assert ('preprocess__description__countvectorizer__analyzer' in
            pipeline.get_params())


def test_set_params():
    pipeline = Pipeline([
        ("preprocess", DataFrameMapper([
            ("description", CountVectorizer()),
        ])),
        ("classify", SVC(kernel='linear'))
    ])
    new_par = {'preprocess__description__countvectorizer__analyzer': 'another'}
    pipeline.set_params(**new_par)
    params = pipeline.get_params()
    for k, v in new_par.items():
        assert params[k] == v