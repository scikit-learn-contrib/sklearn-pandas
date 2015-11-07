__version__ = '0.0.12'

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn import cross_validation
from sklearn import grid_search
import sys

# load in the correct stringtype: str for py3, basestring for py2
string_types = str if sys.version_info >= (3, 0) else basestring


def cross_val_score(model, X, *args, **kwargs):
    X = DataWrapper(X)
    return cross_validation.cross_val_score(model, X, *args, **kwargs)


class GridSearchCV(grid_search.GridSearchCV):
    def fit(self, X, *params, **kwparams):
        super(GridSearchCV, self).fit(DataWrapper(X), *params, **kwparams)

    def predict(self, X, *params, **kwparams):
        super(GridSearchCV, self).fit(DataWrapper(X), *params, **kwparams)


try:
    class RandomizedSearchCV(grid_search.RandomizedSearchCV):
        def fit(self, X, *params, **kwparams):
            super(RandomizedSearchCV, self).fit(DataWrapper(X), *params, **kwparams)

        def predict(self, X, *params, **kwparams):
            super(RandomizedSearchCV, self).fit(DataWrapper(X), *params, **kwparams)
except AttributeError:
    pass


class DataWrapper(object):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, key):
        return self.df.iloc[key]


class PassthroughTransformer(TransformerMixin):
    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X):
        return np.array(X).astype(np.float)


def _handle_feature(fea):
    # convert 1-dimensional arrays to 2-dimensional column vectors
    if len(fea.shape) == 1:
        fea = np.array([fea]).T

    return fea


class DataFrameMapper(BaseEstimator, TransformerMixin):
    """
    Map Pandas data frame column subsets to their own
    sklearn transformation.
    """

    def __init__(self, features):
        """
        Params:

        features    a list of pairs. The first element is the pandas column
                    selector. This can be a string (for one column) or a list
                    of strings. The second element is an object that supports
                    sklearn's transform interface.
        """
        self.features = features

    def _get_col_subset(self, X, cols):
        """
        Get a subset of columns from the given table X.

        X       a Pandas dataframe; the table to select columns from
        cols    a string or list of strings representing the columns
                to select

        Returns a numpy array with the data from the selected columns
        """
        return_vector = False
        if isinstance(cols, string_types):
            return_vector = True
            cols = [cols]

        if isinstance(X, list):
            X = [x[cols] for x in X]
            X = pd.DataFrame(X)

        elif isinstance(X, DataWrapper):
            # if it's a datawrapper, unwrap it
            X = X.df

        if return_vector:
            t = X[cols[0]].values
        else:
            t = X[cols].values

        return t

    def fit(self, X, y=None):
        """
        Fit a transformation from the pipeline

        X       the data to fit
        """
        for columns, transformers in self.features:
            if transformers is not None:
                if isinstance(transformers, list):
                    # first fit_transform all transformers except the last one
                    Xt = self._get_col_subset(X, columns)
                    for transformer in transformers[:-1]:
                        Xt = transformer.fit_transform(Xt)
                    # then fit the last one without transformation
                    transformers[-1].fit(Xt)
                else:
                    transformers.fit(self._get_col_subset(X, columns))
        return self

    def transform(self, X):
        """
        Transform the given data. Assumes that fit has already been called.

        X       the data to transform
        """
        extracted = []
        for columns, transformers in self.features:
            # columns could be a string or list of
            # strings; we don't care because pandas
            # will handle either.
            Xt = self._get_col_subset(X, columns)
            if transformers is not None:
                if isinstance(transformers, list):
                    for transformer in transformers:
                        Xt = transformer.transform(Xt)
                else:
                    Xt = transformers.transform(Xt)
            extracted.append(_handle_feature(Xt))

        # combine the feature outputs into one array.
        # at this point we lose track of which features
        # were created from which input columns, so it's
        # assumed that that doesn't matter to the model.

        # If any of the extracted features is sparse, combine to produce a
        # sparse matrix. Otherwise, produce a dense one.
        if any(sparse.issparse(fea) for fea in extracted):
            stacked = sparse.hstack(extracted).tocsr()
        else:
            stacked = np.hstack(extracted)
        return stacked
