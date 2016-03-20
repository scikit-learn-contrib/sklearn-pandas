import sys
import pandas as pd
import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin

from .cross_validation import DataWrapper
from .pipeline import make_transformer_pipeline

# load in the correct stringtype: str for py3, basestring for py2
string_types = str if sys.version_info >= (3, 0) else basestring


def _handle_feature(fea):
    """
    Convert 1-dimensional arrays to 2-dimensional column vectors.
    """
    if len(fea.shape) == 1:
        fea = np.array([fea]).T

    return fea


def _build_transformer(transformers):
    if isinstance(transformers, list):
        transformers = make_transformer_pipeline(*transformers)
    return transformers


class DataFrameMapper(BaseEstimator, TransformerMixin):
    """
    Map pandas DataFrame column subsets via sklearn transforms to feature
    arrays.

    Parameters
    ----------
        features : list of tuples of the form (column_selector, transform)
            A column selector may be a string (for selecting a single column
            as a 1-d array) or a list of string (for selecting one or more
            columns as a 2-d array).
            A transform is an object which supports sklearns' transform
            interface, or a list of such objects.

        sparse : bool, optional (default=False)
            Return a sparse matrix if set True and any of the extracted
            features are sparse.

    Attributes
    ----------
        feature_indices_ : array of shape (len(self.features) + 1,)
            Indices of self.features in the extracted array.
            Feature ``i`` in self.features is mapped to features from
            ``feature_indices_[i]`` to ``feature_indices_[i+1]`` in transformed
            output.
    """

    def __init__(self, features, sparse=False):
        if isinstance(features, list):
            features = [(columns, _build_transformer(transformers))
                        for (columns, transformers) in features]
        self.features = features
        self.sparse = sparse

    def __setstate__(self, state):
        # compatibility shim for pickles created with sklearn-pandas<1.0.0
        self.features = [(columns, _build_transformer(transformers))
                         for (columns, transformers) in state['features']]
        self.sparse = state.get('sparse', False)

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
                transformers.fit(self._get_col_subset(X, columns))
        return self

    def transform(self, X):
        """
        Transform the given data. Assumes that fit has already been called.

        X       the data to transform
        """
        extracted = []
        self.feature_indices_ = [0]

        for columns, transformers in self.features:
            # columns could be a string or list of
            # strings; we don't care because pandas
            # will handle either.
            Xt = self._get_col_subset(X, columns)
            if transformers is not None:
                Xt = transformers.transform(Xt)

            feature = _handle_feature(Xt)
            extracted.append(feature)
            self.feature_indices_.append(self.feature_indices_[-1] +
                                         feature.shape[1])

        # combine the feature outputs into one array.
        # at this point we lose track of which features
        # were created from which input columns, so it's
        # assumed that that doesn't matter to the model.

        # If any of the extracted features is sparse, combine sparsely.
        # Otherwise, combine as normal arrays.
        if any(sparse.issparse(feature) for feature in extracted):
            stacked = sparse.hstack(extracted).tocsr()
            # return a sparse matrix only if the mapper was initialized
            # with sparse=True
            if not self.sparse:
                stacked = stacked.toarray()
        else:
            stacked = np.hstack(extracted)

        return stacked
