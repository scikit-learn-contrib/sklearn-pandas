import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin


def handle_feature(fea):
    """
    Convert 1-dimensional arrays to 2-dimensional column vectors.
    """
    if len(fea.shape) == 1:
        fea = np.array([fea]).T

    return fea


class PassThroughTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer that just returns the selected column(s) oriented vertically.
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return handle_feature(X)


class ColumnSelectTransformer(BaseEstimator, TransformerMixin):
    """
    A simple Transformer which selects a column or a group of columns from a
    Pandas' DataFrame
    """
    def __init__(self, column_name):
        """
        A Transformer which selects a column or a group of columns from a Pandas' DataFrame
        :param column_name: string or list of strings of columns to select
        """
        self.column_name = column_name

    def fit(self, X, y=None):
        if not (isinstance(X, pd.DataFrame) or isinstance(X, pd.Series)):
            raise TypeError('Input should be a Pandas DataFrame or a Series (was %s)' % type(X))
        column_name = self.column_name
        # in case in bracketed as [] to output a (n,1) rather (n,) shape
        if not isinstance(column_name, list):
            column_name = [column_name]
        for name in column_name:
            if name not in X.columns:
                raise ValueError('Select column name %s is not in %s' % (name, X.columns))
        return self

    def transform(self, X, y=None):
        return X[self.column_name]
