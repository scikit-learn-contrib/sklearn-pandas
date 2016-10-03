import pandas as pd
import numpy as np

import pytest

from sklearn_pandas.utils import (
    ColumnSelectTransformer, PassThroughTransformer, handle_feature)


@pytest.fixture
def df():
    return pd.DataFrame({'a': [1, 2], 'b': [3, 4]})


def test_column_select_1d(df):
    cst = ColumnSelectTransformer('a')
    selected = cst.fit_transform(df)
    assert (selected == np.array([1, 2])).all()


def test_column_select_2d(df):
    cst = ColumnSelectTransformer(['a'])
    selected = cst.fit_transform(df)
    assert (selected == np.array([[1], [2]])).all()


def test_column_select_nonframe():
    """
    ColumnSelectTransformer only works with Series or DataFrames.
    """
    cst = ColumnSelectTransformer('a')
    with pytest.raises(TypeError):
        cst.fit_transform({})


def test_column_select_nonexistent(df):
    """
    Trying to select an unexistent column raises ValueError.
    """
    cst = ColumnSelectTransformer(['z'])
    with pytest.raises(ValueError):
        selected = cst.fit_transform(df)


def test_passthrough_transformer(df):
    pt = PassThroughTransformer()
    result = pt.fit_transform(np.array([1, 2]))
    assert (result == np.array([[1], [2]])).all()


def test_handle_feature():
    feature = np.array([1, 2, 3])
    assert handle_feature(feature).shape == (3, 1)

    feature = np.array([[1], [2], [3]])
    assert handle_feature(feature).shape == (3, 1)