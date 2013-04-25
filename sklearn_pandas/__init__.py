
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn import cross_validation
import pdb

def cross_val_score(estimator, X, *args, **kwargs):
    class DataFrameWrapper(object):
        def __init__(self, df):
            self.df = df

        def __eq__(self, other):
            return self.df is other.df

    class DataFrameMapper(BaseEstimator):
        def __init__(self, estimator, X):
            self.estimator = estimator
            self.X = X

        def fit(self, x, y):
            self.estimator.fit(self._get_row_subset(x), y)
            return self

        def transform(self, x):
            return self.estimator.transform(self._get_row_subset(x))

        def predict(self, x):
            return self.estimator.predict(self._get_row_subset(x))

        def _get_row_subset(self, rows):
            subset = self.X.df.irow(rows)
            subset.index = range(0, len(rows))
            return subset
    
    X_indices = range(len(X))
    X_wrapped = DataFrameWrapper(X)
    df = DataFrameMapper(estimator, X_wrapped)
    return cross_validation.cross_val_score(df, X_indices, *args, **kwargs)


class DataFrameMapper(BaseEstimator, TransformerMixin):
    def __init__(self, features):
        self.features = features

    def fit(self, X, y=None):
        for columns, transformer in self.features:
            try:
                transformer.fit(X[columns], y)
            except TypeError:
                transformer.fit(X[columns])
        return self

    def transform(self, X):
        extracted = []
        for columns, transformer in self.features:
            fea = transformer.transform(X[columns])
            if hasattr(fea, "toarray"):
                extracted.append(fea.toarray())
            else:
                if len(fea.shape) == 1:
                    fea = np.array([fea]).T
                extracted.append(fea)
        if len(extracted) > 1:
            return np.concatenate(extracted, axis=1)
        else: 
            return extracted[0]

