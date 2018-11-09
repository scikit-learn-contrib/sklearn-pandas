# -*- coding: utf8 -*-

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
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import (
    Imputer, StandardScaler, OneHotEncoder, LabelBinarizer, LabelEncoder)
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


class DateEncoder():
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        dt = X.dt
        return pd.concat([dt.year, dt.month, dt.day], axis=1)


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
    return pd.DataFrame({'target': ['a', 'a', 'b', 'b', 'c', 'c'],
                         'feat1': [1, 2, 3, 4, 5, 6],
                         'feat2': [1, 2, 3, 2, 3, 4]})


@pytest.fixture
def multiindex_dataframe():
    """Example MultiIndex DataFrame, taken from pandas documentation
    """
    iterables = [['bar', 'baz', 'foo', 'qux'], ['one', 'two']]
    index = pd.MultiIndex.from_product(iterables, names=['first', 'second'])
    df = pd.DataFrame(np.random.randn(10, 8), columns=index)
    return df


@pytest.fixture
def multiindex_dataframe_incomplete(multiindex_dataframe):
    """Example MultiIndex DataFrame with missing entries
    """
    df = multiindex_dataframe
    mask_array = np.zeros(df.size)
    mask_array[:20] = 1
    np.random.shuffle(mask_array)
    mask = mask_array.reshape(df.shape).astype(bool)
    df.mask(mask, inplace=True)
    return df


def test_transformed_names_simple(simple_dataframe):
    """
    Get transformed names of features in `transformed_names` attribute
    for simple transformation
    """
    df = simple_dataframe
    mapper = DataFrameMapper([('a', None)])
    mapper.fit_transform(df)
    assert mapper.transformed_names_ == ['a']


def test_transformed_names_binarizer(complex_dataframe):
    """
    Get transformed names of features in `transformed_names` attribute
    for a transformation that multiplies the number of columns
    """
    df = complex_dataframe
    mapper = DataFrameMapper([('target', LabelBinarizer())])
    mapper.fit_transform(df)
    assert mapper.transformed_names_ == ['target_a', 'target_b', 'target_c']


def test_transformed_names_binarizer_unicode():
    df = pd.DataFrame({'target': [u'ñ', u'á', u'é']})
    mapper = DataFrameMapper([('target', LabelBinarizer())])
    mapper.fit_transform(df)
    expected_names = {u'target_ñ', u'target_á', u'target_é'}
    assert set(mapper.transformed_names_) == expected_names


def test_transformed_names_transformers_list(complex_dataframe):
    """
    When using a list of transformers, use them in inverse order to get the
    transformed names
    """
    df = complex_dataframe
    mapper = DataFrameMapper([
        ('target', [LabelBinarizer(), MockXTransformer()])
    ])
    mapper.fit_transform(df)
    assert mapper.transformed_names_ == ['target_a', 'target_b', 'target_c']


def test_transformed_names_simple_alias(simple_dataframe):
    """
    If we specify an alias for a single output column, it is used for the
    output
    """
    df = simple_dataframe
    mapper = DataFrameMapper([('a', None, {'alias': 'new_name'})])
    mapper.fit_transform(df)
    assert mapper.transformed_names_ == ['new_name']


def test_transformed_names_complex_alias(complex_dataframe):
    """
    If we specify an alias for a multiple output column, it is used for the
    output
    """
    df = complex_dataframe
    mapper = DataFrameMapper([('target', LabelBinarizer(), {'alias': 'new'})])
    mapper.fit_transform(df)
    assert mapper.transformed_names_ == ['new_a', 'new_b', 'new_c']


def test_exception_column_context_transform(simple_dataframe):
    """
    If an exception is raised when transforming a column,
    the exception includes the name of the column being transformed
    """
    class FailingTransformer(object):
        def fit(self, X):
            pass

        def transform(self, X):
            raise Exception('Some exception')

    df = simple_dataframe
    mapper = DataFrameMapper([('a', FailingTransformer())])
    mapper.fit(df)

    with pytest.raises(Exception, match='a: Some exception'):
        mapper.transform(df)


def test_exception_column_context_fit(simple_dataframe):
    """
    If an exception is raised when fit a column,
    the exception includes the name of the column being fitted
    """
    class FailingFitter(object):
        def fit(self, X):
            raise Exception('Some exception')

    df = simple_dataframe
    mapper = DataFrameMapper([('a', FailingFitter())])

    with pytest.raises(Exception, match='a: Some exception'):
        mapper.fit(df)


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


def test_numeric_column_names(complex_dataframe):
    """
    Get a dataframe from a complex mapped dataframe with numeric column names
    """
    df = complex_dataframe
    df.columns = [0, 1, 2]
    mapper = DataFrameMapper(
        [(0, None), (1, None), (2, None)], df_out=True)
    transformed = mapper.fit_transform(df)
    assert len(transformed) == len(complex_dataframe)
    for c in df.columns:
        assert len(transformed[c]) == len(df[c])


def test_multiindex_df(multiindex_dataframe_incomplete):
    """
    Get a dataframe from a multiindex dataframe with missing data
    """
    df = multiindex_dataframe_incomplete
    mapper = DataFrameMapper([([c], Imputer()) for c in df.columns],
                             df_out=True)
    transformed = mapper.fit_transform(df)
    assert len(transformed) == len(multiindex_dataframe_incomplete)
    for c in df.columns:
        assert len(transformed[str(c)]) == len(df[c])


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


def test_preserve_df_index():
    """
    The index is preserved when df_out=True
    """
    df = pd.DataFrame({'target': [1, 2, 3]},
                      index=['a', 'b', 'c'])
    mapper = DataFrameMapper([('target', None)],
                             df_out=True)

    transformed = mapper.fit_transform(df)

    assert_array_equal(transformed.index, df.index)


def test_preserve_df_index_rows_dropped():
    """
    If df_out=True but the original df index length doesn't
    match the number of final rows, use a numeric index
    """
    class DropLastRowTransformer(object):
        def fit(self, X):
            return self

        def transform(self, X):
            return X[:-1]

    df = pd.DataFrame({'target': [1, 2, 3]},
                      index=['a', 'b', 'c'])
    mapper = DataFrameMapper([('target', DropLastRowTransformer())],
                             df_out=True)

    transformed = mapper.fit_transform(df)

    assert_array_equal(transformed.index, np.array([0, 1]))


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


def test_fit_transform(simple_dataframe):
    """
    Check that custom fit_transform methods of the transformers are invoked.
    """
    df = simple_dataframe
    mock_transformer = Mock()
    # return something of measurable length but does nothing
    mock_transformer.fit_transform.return_value = np.array([1, 2, 3])
    mapper = DataFrameMapper([("a", mock_transformer)])
    mapper.fit_transform(df)
    assert mock_transformer.fit_transform.called


def test_fit_transform_equiv_mock(simple_dataframe):
    """
    Check for equivalent results for code paths fit_transform
    versus fit and transform in DataFrameMapper using the mock
    transformer which does not implement a custom fit_transform.
    """
    df = simple_dataframe
    mapper = DataFrameMapper([('a', MockXTransformer())])
    transformed_combined = mapper.fit_transform(df)
    transformed_separate = mapper.fit(df).transform(df)
    assert np.all(transformed_combined == transformed_separate)


def test_fit_transform_equiv_pca(complex_dataframe):
    """
    Check for equivalent results for code paths fit_transform
    versus fit and transform in DataFrameMapper and transformer
    using PCA which implements a custom fit_transform. The
    equivalence of both paths in the transformer only can be
    asserted since this is tested in the sklearn tests
    scikit-learn/sklearn/decomposition/tests/test_pca.py
    """
    df = complex_dataframe
    mapper = DataFrameMapper(
        [(['feat1', 'feat2'], sklearn.decomposition.PCA(2))],
        df_out=True)
    transformed_combined = mapper.fit_transform(df)
    transformed_separate = mapper.fit(df).transform(df)
    assert np.allclose(transformed_combined, transformed_separate)


def test_input_df_true_first_transformer(simple_dataframe, monkeypatch):
    """
    If input_df is True, the first transformer is passed
    a pd.Series instead of an np.array
    """
    df = simple_dataframe
    monkeypatch.setattr(MockXTransformer, 'fit', Mock())
    monkeypatch.setattr(MockXTransformer, 'transform',
                        Mock(return_value=np.array([1, 2, 3])))
    mapper = DataFrameMapper([
        ('a', MockXTransformer())
    ], input_df=True)
    out = mapper.fit_transform(df)

    args, _ = MockXTransformer().fit.call_args
    assert isinstance(args[0], pd.Series)

    args, _ = MockXTransformer().transform.call_args
    assert isinstance(args[0], pd.Series)

    assert_array_equal(out, np.array([1, 2, 3]).reshape(-1, 1))


def test_input_df_true_next_transformers(simple_dataframe, monkeypatch):
    """
    If input_df is True, the subsequent transformers get passed pandas
    objects instead of numpy arrays (given the previous transformers
    output pandas objects as well)
    """
    df = simple_dataframe
    monkeypatch.setattr(MockTClassifier, 'fit', Mock())
    monkeypatch.setattr(MockTClassifier, 'transform',
                        Mock(return_value=pd.Series([1, 2, 3])))
    mapper = DataFrameMapper([
        ('a', [MockXTransformer(), MockTClassifier()])
    ], input_df=True)
    mapper.fit(df)
    out = mapper.transform(df)

    args, _ = MockTClassifier().fit.call_args
    assert isinstance(args[0], pd.Series)

    assert_array_equal(out, np.array([1, 2, 3]).reshape(-1, 1))


def test_input_df_true_multiple_cols(complex_dataframe):
    """
    When input_df is True, applying transformers to multiple columns
    works as expected
    """
    df = complex_dataframe

    mapper = DataFrameMapper([
        ('target', MockXTransformer()),
        ('feat1',  MockXTransformer()),
    ], input_df=True)
    out = mapper.fit_transform(df)

    assert_array_equal(out[:, 0], df['target'].values)
    assert_array_equal(out[:, 1], df['feat1'].values)


def test_input_df_date_encoder():
    """
    When input_df is True we can apply a transformer that only works
    with pandas dataframes like a DateEncoder
    """
    df = pd.DataFrame(
        {'dates': pd.date_range('2015-10-30', '2015-11-02')})
    mapper = DataFrameMapper([
        ('dates', DateEncoder())
    ], input_df=True)
    out = mapper.fit_transform(df)
    expected = np.array([
        [2015, 10, 30],
        [2015, 10, 31],
        [2015, 11, 1],
        [2015, 11, 2]
    ])
    assert_array_equal(out, expected)


def test_local_input_df_date_encoder():
    """
    When input_df is True we can apply a transformer that only works
    with pandas dataframes like a DateEncoder
    """
    df = pd.DataFrame(
        {'dates': pd.date_range('2015-10-30', '2015-11-02')})
    mapper = DataFrameMapper([
        ('dates', DateEncoder(), {'input_df': True})
    ], input_df=False)
    out = mapper.fit_transform(df)
    expected = np.array([
        [2015, 10, 30],
        [2015, 10, 31],
        [2015, 11, 1],
        [2015, 11, 2]
    ])
    assert_array_equal(out, expected)


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
    If a string is specified as the columns, the transformer
    is called with a 1-d array as input.
    """
    df = simple_dataframe
    mock_transformer = Mock()
    mapper = DataFrameMapper([("a", mock_transformer)])

    mapper.fit(df)
    args, kwargs = mock_transformer.fit.call_args
    assert args[0].shape == (3,)


def test_cols_list_column_vector(simple_dataframe):
    """
    If a one-element list is specified as the columns, the transformer
    is called with a column vector as input.
    """
    df = simple_dataframe
    mock_transformer = Mock()
    mapper = DataFrameMapper([(["a"], mock_transformer)])

    mapper.fit(df)
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


def test_default_none_names():
    """
    If default=None, column names are returned unmodified.
    """
    df = pd.DataFrame({'a': [1, 2, 3], 'b': [3, 5, 7]})
    mapper = DataFrameMapper([], default=None)

    mapper.fit_transform(df)
    assert mapper.transformed_names_ == ['a', 'b']


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
    dataframe = pd.DataFrame({"a": [1, np.nan, 3], "b": [1, 5, 7]},
                             dtype=np.float64)

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


def test_build_features_old_unpickle(simple_dataframe):
    """
    Fitted mappers pickled before the built_features and built_default
    attributes can correctly transform
    """
    df = simple_dataframe
    mapper = DataFrameMapper([('a', None)])
    mapper.fit(df)

    # simulate the mapper was pickled before the attributes existed
    del mapper.built_features
    del mapper.built_default

    mapper_pickled = pickle.dumps(mapper)
    loaded_mapper = pickle.loads(mapper_pickled)
    loaded_mapper.transform(simple_dataframe)  # doesn't fail


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


def test_dict_vectorizer():
    df = pd.DataFrame(
        [[{'a': 1, 'b': 2}], [{'a': 3}]],
        columns=['colA']
    )

    outdf = DataFrameMapper(
        [('colA', DictVectorizer())],
        df_out=True,
        default=False
    ).fit_transform(df)

    columns = sorted(list(outdf.columns))
    assert len(columns) == 2
    assert columns[0] == 'colA_a'
    assert columns[1] == 'colA_b'


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


def test_heterogeneous_output_types_input_df():
    """
    Modify feat2, but pass feat1 through unmodified.
    This fails if input_df == False
    """
    df = pd.DataFrame({
        'feat1': [1, 2, 3, 4, 5, 6],
        'feat2': [1.0, 2.0, 3.0, 2.0, 3.0, 4.0]
    })
    M = DataFrameMapper([
        (['feat2'], StandardScaler())
        ], input_df=True, df_out=True, default=None)
    dft = M.fit_transform(df)
    assert dft['feat1'].dtype == np.dtype('int64')
    assert dft['feat2'].dtype == np.dtype('float64')


def test_inverse_transform_simple():
    df = pd.DataFrame({'colA': list('ynyyn'), 'colB': list('abcab')})
    mapper = DataFrameMapper([
        ('colA', LabelEncoder()),
        ('colB', LabelEncoder()),
    ])

    transformed = mapper.fit_transform(df)
    restored = mapper.inverse_transform(transformed)

    assert isinstance(restored, pd.DataFrame)
    assert restored.equals(df)


def test_inverse_transform_multicolumn():
    df = pd.DataFrame({'colA': list('ynyyn'),
                       'colB': list('abcab'),
                       'colC': list('sttts')})
    mapper = DataFrameMapper([
        ('colA', LabelEncoder()),
        ('colB', LabelBinarizer()),
        ('colC', LabelEncoder()),
    ])

    transformed = mapper.fit_transform(df)
    restored = mapper.inverse_transform(transformed)

    assert isinstance(restored, pd.DataFrame)
    assert restored.equals(df)
