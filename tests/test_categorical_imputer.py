import pytest

import numpy as np
import pandas as pd

from sklearn_pandas import CategoricalImputer
from sklearn_pandas import DataFrameMapper

# In sklearn18 NotFittedError was moved from utils.validation
# to exceptions module.
try:
    from sklearn.exceptions import NotFittedError
except ImportError:
    from sklearn.utils.validation import NotFittedError


@pytest.mark.parametrize('none_value', [None, np.nan])
@pytest.mark.parametrize('input_type', ['np', 'pd'])
def test_unit(input_type, none_value):

    data = ['a', 'b', 'b', none_value]

    if input_type == 'pd':
        X = pd.Series(data)
    else:
        X = np.asarray(data, dtype=object)

    Xc = X.copy()

    Xt = CategoricalImputer().fit_transform(X)

    assert (np.asarray(X) == np.asarray(Xc)).all()
    assert isinstance(Xt, np.ndarray)
    assert (Xt == ['a', 'b', 'b', 'b']).all()


@pytest.mark.parametrize('input_type', ['np', 'pd'])
def test_no_mode(input_type):

    data = ['a', 'b', 'c', np.nan]

    if input_type == 'pd':
        X = pd.Series(data)
    else:
        X = np.asarray(data, dtype=object)

    with pytest.raises(ValueError):
        CategoricalImputer().fit_transform(X)


@pytest.mark.parametrize('input_type', ['np', 'pd'])
def test_missing_values_param(input_type):

    data = ['x', 'y', 'a_missing', 'y']

    if input_type == 'pd':
        X = pd.Series(data)
    else:
        X = np.asarray(data, dtype=object)

    imp = CategoricalImputer(missing_values='a_missing')
    Xt = imp.fit_transform(X)

    assert (Xt == np.array(['x', 'y', 'y', 'y'])).all()


@pytest.mark.parametrize('input_type', ['np', 'pd'])
def test_copy_param(input_type):

    data = ['a', np.nan, 'b', 'a']

    if input_type == 'pd':
        X = pd.Series(data)
    else:
        X = np.asarray(data, dtype=object)

    imp = CategoricalImputer(copy=False)
    Xt = imp.fit_transform(X)

    Xe = np.array(['a', 'a', 'b', 'a'])
    assert (Xt == Xe).all()
    assert (X == Xe).all()


@pytest.mark.parametrize('input_type', ['np', 'pd'])
def test_data_type(input_type):

    data = ['a', np.nan, 'b', 3, 'a', 3, 'a', 4.5]

    if input_type == 'pd':
        X = pd.Series(data)
    else:
        X = np.asarray(data, dtype=object)

    Xt = CategoricalImputer().fit_transform(X)

    Xe = np.array(['a', 'a', 'b', 3, 'a', 3, 'a', 4.5], dtype=object)
    assert (Xt == Xe).all()


@pytest.mark.parametrize('none_value', [None, np.nan])
def test_integration(none_value):

    df = pd.DataFrame({'cat': ['a', 'a', 'a', none_value, 'b'],
                       'num': [1, 2, 3, 4, 5]})

    mapper = DataFrameMapper([
        ('cat', CategoricalImputer()),
        ('num', None)
    ], df_out=True).fit(df)

    df_t = mapper.transform(df)

    assert pd.notnull(df_t).all().all()

    val_idx = pd.notnull(df['cat'])
    nan_idx = ~val_idx

    assert (df['num'] == df_t['num']).all()

    assert (df['cat'][val_idx] == df_t['cat'][val_idx]).all()
    assert (df_t['cat'][nan_idx] == df['cat'].mode().values[0]).all()


def test_not_fitted():
    """
    If imputer is not fitted, NotFittedError is raised.
    """
    imp = CategoricalImputer()
    with pytest.raises(NotFittedError):
        imp.transform(np.array(['a', 'b', 'b', None]))


@pytest.mark.parametrize('input_type', ['np', 'pd'])
@pytest.mark.parametrize('replacement_value', ['a', 'c'])
def test_custom_replacement(replacement_value, input_type):
    """
    If replacement != 'mode', impute with that value instead of mode
    """
    data = ['a', np.nan, 'b', 'b']

    if input_type == 'pd':
        X = pd.Series(data)
    else:
        X = np.asarray(data, dtype=object)

    Xc = X.copy()

    Xt = CategoricalImputer(
        strategy='fixed_value',
        replacement=replacement_value
    ).fit_transform(X)

    assert (np.asarray(X) == np.asarray(Xc)).all()
    assert isinstance(Xt, np.ndarray)
    assert (Xt == ['a', replacement_value, 'b', 'b']).all()


def test_missing_replacement():
    """
    Raise error if no replacement value specified and strategy='fixed_value'
    """
    with pytest.raises(ValueError):
        CategoricalImputer(strategy="fixed_value")


def test_invalid_strategy():
    """
    Raise an error if an invalid strategy is entered
    """
    with pytest.raises(ValueError):
        CategoricalImputer(strategy="not_a_supported_strategy")
