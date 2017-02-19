import pytest

import numpy as np
import pandas as pd

from sklearn_pandas import DataFrameImputer


def test_dataframe_imputer():

    data = [
        ['a', 1, 2],
        ['b', 1, 1],
        ['b', 2, 2],
        [np.nan, np.nan, np.nan]
    ]

    X = pd.DataFrame(data)
    xt = DataFrameImputer().fit_transform(X)

    assert type(xt) == pd.DataFrame
    assert xt.equals(xt.dropna())
    assert not X.equals(xt)
