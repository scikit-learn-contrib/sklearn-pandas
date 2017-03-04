import pytest

import numpy as np
import pandas as pd

from sklearn_pandas import CategoricalImputer
from sklearn_pandas import DataFrameMapper


@pytest.mark.parametrize('none_value', [None, np.nan])
@pytest.mark.parametrize('input_type', ['np', 'pd'])
def test_unit(input_type, none_value):

    data = ['a', 'b', 'b', none_value]

    if input_type == 'pd':
        X = pd.Series(data)
    else:
        X = np.asarray(data)
    
    Xc = X.copy()

    Xt = CategoricalImputer().fit_transform(X)

    assert (np.asarray(X) == np.asarray(Xc)).all()
    assert type(Xt) == np.ndarray
    assert len(X) == len(Xt)
    assert len(Xt[pd.isnull(Xt)]) == 0

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
