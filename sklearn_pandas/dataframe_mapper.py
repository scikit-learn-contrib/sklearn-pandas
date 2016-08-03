import sys
import pandas as pd
import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin

from .cross_validation import DataWrapper
from .pipeline import make_transformer_pipeline, _call_fit

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
    Map Pandas data frame column subsets to their own
    sklearn transformation.
    """

    def __init__(self, features, default=False, sparse=False):
        """
        Params:

        features    a list of pairs. The first element is the pandas column
                    selector. This can be a string (for one column) or a list
                    of strings. The second element is an object that supports
                    sklearn's transform interface, or a list of such objects.

        default     default transformer to apply to the columns not
                    explicitly selected in the mapper. If False (default),
                    discard them. If None, pass them through untouched. Any
                    other transformer will be applied to all the unselected
                    columns as a whole, taken as a 2d-array.

        sparse      will return sparse matrix if set True and any of the
                    extracted features is sparse. Defaults to False.
        """
        if isinstance(features, list):
            features = [(columns, _build_transformer(transformers))
                        for (columns, transformers) in features]
        self.features = features
        self.default = _build_transformer(default)
        self.sparse = sparse

    @property
    def _selected_columns(self):
        """
        Return a set of selected columns in the feature list.
        """
        selected_columns = set()
        for feature in self.features:
            columns = feature[0]
            if isinstance(columns, list):
                selected_columns = selected_columns.union(set(columns))
            else:
                selected_columns.add(columns)
        return selected_columns

    def _unselected_columns(self, X):
        """
        Return list of columns present in X and not selected explicitly in the
        mapper.

        Unselected columns are returned in the order they appear in the
        dataframe to avoid issues with different ordering during default fit
        and transform steps.
        """
        X_columns = list(X.columns)
        return [column for column in X_columns if
                column not in self._selected_columns]

    def __setstate__(self, state):
        # compatibility shim for pickles created with sklearn-pandas<1.0.0
        self.features = [(columns, _build_transformer(transformers))
                         for (columns, transformers) in state['features']]
        self.sparse = state.get('sparse', False)

        # compatibility shim for pickles created before ``default`` init
        # argument existed
        self.default = state.get('default', False)

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

        y       the target vector relative to X, optional

        """
        for columns, transformers in self.features:
            if transformers is not None:
                _call_fit(transformers.fit,
                          self._get_col_subset(X, columns), y)

        # handle features not explicitly selected
        if self.default:  # not False and not None
            _call_fit(self.default.fit,
                      self._get_col_subset(X, self._unselected_columns(X)), y)
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
                Xt = transformers.transform(Xt)
            extracted.append(_handle_feature(Xt))

        # handle features not explicitly selected
        if self.default is not False:
            Xt = self._get_col_subset(X, self._unselected_columns(X))
            if self.default is not None:
                Xt = self.default.transform(Xt)
            extracted.append(_handle_feature(Xt))


        # combine the feature outputs into one array.
        # at this point we lose track of which features
        # were created from which input columns, so it's
        # assumed that that doesn't matter to the model.

        # If any of the extracted features is sparse, combine sparsely.
        # Otherwise, combine as normal arrays.
        if any(sparse.issparse(fea) for fea in extracted):
            stacked = sparse.hstack(extracted).tocsr()
            # return a sparse matrix only if the mapper was initialized
            # with sparse=True
            if not self.sparse:
                stacked = stacked.toarray()
        else:
            stacked = np.hstack(extracted)

        return stacked
