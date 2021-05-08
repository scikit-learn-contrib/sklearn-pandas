import pytest
from unittest.mock import Mock
import numpy as np
import pandas as pd
from sklearn_pandas import DataFrameMapper
from sklearn.compose import make_column_selector
from sklearn.preprocessing import StandardScaler


class GetStartWith:
    def __init__(self, start_str):
        self.start_str = start_str

    def __call__(self, X: pd.DataFrame) -> list:
        return [c for c in X.columns if c.startswith(self.start_str)]


df = pd.DataFrame({
    'sepal length (cm)': [1.0, 2.0, 3.0],
    'sepal width (cm)': [1.0, 2.0, 3.0],
    'petal length (cm)': [1.0, 2.0, 3.0],
    'petal width (cm)': [1.0, 2.0, 3.0]
})
t = DataFrameMapper([
    (make_column_selector(dtype_include=float), StandardScaler(), {'alias': 'x'}),
    (GetStartWith('petal'), None, {'alias': 'petal'})
], df_out=True, default=False)

t.fit(df)
print(t.transform(df).shape)
