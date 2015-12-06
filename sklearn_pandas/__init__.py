__version__ = '1.0.0'

import numpy as np
from sklearn.base import TransformerMixin

from .dataframe_mapper import DataFrameMapper  # NOQA
from .cross_validation import cross_val_score, GridSearchCV, RandomizedSearchCV  # NOQA


class PassthroughTransformer(TransformerMixin):
    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X):
        return np.array(X).astype(np.float)
