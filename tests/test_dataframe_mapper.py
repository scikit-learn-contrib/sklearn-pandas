import pytest
from pkg_resources import parse_version

# In py3, mock is included with the unittest standard library
# In py2, it's a separate package
try:
    from unittest.mock import Mock
except ImportError:
    from mock import Mock

from pandas import DataFrame
import pandas as pd
from scipy import sparse
from sklearn import __version__ as sklearn_version
from sklearn.cross_validation import cross_val_score as sklearn_cv_score
from sklearn.datasets import load_iris
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import (
    Imputer, StandardScaler, OneHotEncoder, LabelBinarizer)
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.base import BaseEstimator, TransformerMixin
import sklearn.decomposition
import numpy as np
from numpy.testing import assert_array_equal
import pickle

from sklearn_pandas import DataFrameMapper, cross_val_score
from sklearn_pandas.dataframe_mapper import _handle_feature, _build_transformer
from sklearn_pandas.pipeline import TransformerPipeline


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


class CustomTransformer(BaseEstimator, TransformerMixin):
    """
    Example of transformer in which the number of classes
    is not equals to the number of output columns.
    """
    def fit(self, X, y=None):
        self.min = X.min()
        self.classes_ = np.unique(X)
        return self

    def transform(self, X):
        classes = np.unique(X)
        if len(np.setdiff1d(classes, self.classes_)) > 0:
            raise ValueError('Unknown values found.')
        return X - self.min


@pytest.fixture
def simple_dataframe():
    return pd.DataFrame({'a': [1, 2, 3]})


@pytest.fixture
def complex_dataframe():
    return pd.DataFrame({'target': ['a', 'a', 'a', 'b', 'b', 'b'],
                         'feat1': [1, 2, 3, 4, 5, 6],
                         'feat2': [1, 2, 3, 2, 3, 4]})


def test_simple_df(simple_dataframe):
    """
    Get a dataframe from a simple mapped dataframe
    """
    df = simple_dataframe
    mapper = DataFrameMapper([('a', None)], df_out=True)
    transformed = mapper.fit_transform(df)
    assert type(transformed) == pd.DataFrame
    assert len(transformed["a"]) == len(simple_dataframe["a"])


def test_complex_df(complex_dataframe):
    """
    Get a dataframe from a complex mapped dataframe
    """
    df = complex_dataframe
    mapper = DataFrameMapper(
        [('target', None), ('feat1', None), ('feat2', None)],
        df_out=True)
    transformed = mapper.fit_transform(df)
    assert len(transformed) == len(complex_dataframe)
    for c in df.columns:
        assert len(transformed[c]) == len(df[c])


def test_binarizer_df():
    """
    Check level names from LabelBinarizer
    """
    df = pd.DataFrame({'target': ['a', 'a', 'b', 'b', 'c', 'a']})
    mapper = DataFrameMapper([('target', LabelBinarizer())], df_out=True)
    transformed = mapper.fit_transform(df)
    cols = transformed.columns
    assert len(cols) == 3
    assert cols[0] == 'target_a'
    assert cols[1] == 'target_b'
    assert cols[2] == 'target_c'


def test_binarizer_int_df():
    """
    Check level names from LabelBinarizer for a numeric array.
    """
    df = pd.DataFrame({'target': [5, 5, 6, 6, 7, 5]})
    mapper = DataFrameMapper([('target', LabelBinarizer())], df_out=True)
    transformed = mapper.fit_transform(df)
    cols = transformed.columns
    assert len(cols) == 3
    assert cols[0] == 'target_5'
    assert cols[1] == 'target_6'
    assert cols[2] == 'target_7'


def test_binarizer2_df():
    """
    Check level names from LabelBinarizer with just one output column
    """
    df = pd.DataFrame({'target': ['a', 'a', 'b', 'b', 'a']})
    mapper = DataFrameMapper([('target', LabelBinarizer())], df_out=True)
    transformed = mapper.fit_transform(df)
    cols = transformed.columns
    assert len(cols) == 1
    assert cols[0] == 'target'


def test_onehot_df():
    """
    Check level ids from one-hot
    """
    df = pd.DataFrame({'target': [0, 0, 1, 1, 2, 3, 0]})
    mapper = DataFrameMapper([(['target'], OneHotEncoder())], df_out=True)
    transformed = mapper.fit_transform(df)
    cols = transformed.columns
    assert len(cols) == 4
    assert cols[0] == 'target_0'
    assert cols[3] == 'target_3'


def test_customtransform_df():
    """
    Check level ids from a transformer in which
    the number of classes is not equals to the number of output columns.
    """
    df = pd.DataFrame({'target': [6, 5, 7, 5, 4, 8, 8]})
    mapper = DataFrameMapper([(['target'], CustomTransformer())], df_out=True)
    transformed = mapper.fit_transform(df)
    cols = transformed.columns
    assert len(mapper.features[0][1].classes_) == 5
    assert len(cols) == 1
    assert cols[0] == 'target'


def test_pca(complex_dataframe):
    """
    Check multi in and out with PCA
    """
    df = complex_dataframe
    mapper = DataFrameMapper(
        [(['feat1', 'feat2'], sklearn.decomposition.PCA(2))],
        df_out=True)
    transformed = mapper.fit_transform(df)
    cols = transformed.columns
    assert len(cols) == 2
    assert cols[0] == 'feat1_feat2_0'
    assert cols[1] == 'feat1_feat2_1'


def test_nonexistent_columns_explicit_fail(simple_dataframe):
    """
    If a nonexistent column is selected, KeyError is raised.
    """
    mapper = DataFrameMapper(None)
    with pytest.raises(KeyError):
        mapper._get_col_subset(simple_dataframe, ['nonexistent_feature'])


def test_get_col_subset_single_column_array(simple_dataframe):
    """
    Selecting a single column should return a 1-dimensional numpy array.
    """
    mapper = DataFrameMapper(None)
    array = mapper._get_col_subset(simple_dataframe, "a")

    assert type(array) == np.ndarray
    assert array.shape == (len(simple_dataframe["a"]),)


def test_get_col_subset_single_column_list(simple_dataframe):
    """
    Selecting a list of columns (even if the list contains a single element)
    should return a 2-dimensional numpy array.
    """
    mapper = DataFrameMapper(None)
    array = mapper._get_col_subset(simple_dataframe, ["a"])

    assert type(array) == np.ndarray
    assert array.shape == (len(simple_dataframe["a"]), 1)


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
    assert_array_equal(_handle_feature(array), array)


def test_handle_feature_1dim():
    """
    1-dimensional arrays are converted to 2-dimensional column vectors.
    """
    array = np.array([1, 2])
    assert_array_equal(_handle_feature(array), np.array([[1], [2]]))


def test_build_transformers():
    """
    When a list of transformers is passed, return a pipeline with
    each element of the iterable as a step of the pipeline.
    """
    transformers = [MockTClassifier(), MockTClassifier()]
    pipeline = _build_transformer(transformers)
    assert isinstance(pipeline, Pipeline)
    for ix, transformer in enumerate(transformers):
        assert pipeline.steps[ix][1] == transformer


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
    assert (transformed[: 0] == np.array([1., 2., 3.])).all()


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
    mapper = DataFrameMapper(None)
    # simulate the mapper was created with < 1.0.0 code
    mapper.features = [('a', [MockXTransformer()])]
    mapper_pickled = pickle.dumps(mapper)

    loaded_mapper = pickle.loads(mapper_pickled)
    transformer = loaded_mapper.features[0][1]
    assert isinstance(transformer, TransformerPipeline)
    assert isinstance(transformer.steps[0][1], MockXTransformer)


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


@pytest.mark.skipIf(parse_version(sklearn_version) < parse_version('0.16'))
def test_direct_cross_validation(iris_dataframe):
    """
    Starting with sklearn>=0.16.0 we no longer need CV wrappers for dataframes.
    See https://github.com/paulgb/sklearn-pandas/issues/11
    """
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
    scores = sklearn_cv_score(pipeline, data, labels)
    assert scores.mean() > 0.96
    assert (scores.std() * 2) < 0.04
