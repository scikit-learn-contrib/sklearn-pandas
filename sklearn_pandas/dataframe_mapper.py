import sys
import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin

from .pipeline import make_feature_union
from .utils import handle_feature

# load in the correct stringtype: str for py3, basestring for py2
string_types = str if sys.version_info >= (3, 0) else basestring


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
        self.pipeline = make_feature_union(features)
        self.features = features
        self.default = default
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
        self.features = state['features']

        # compatibility for pickles before FeatureUnion
        self.pipeline = state.get('pipeline',
                                  make_feature_union(state['features']))

        # compatibility shim for pickles created with sklearn-pandas<1.0.0
        self.sparse = state.get('sparse', False)

        # compatibility shim for pickles created before ``default`` init
        # argument existed
        self.default = state.get('default', False)

    def fit(self, X, y=None):
        """
        Fit a transformation from the pipeline

        X       the data to fit

        y       the target vector relative to X, optional

        """
        if self.pipeline is not None:
            self.pipeline.fit(X, y)

        # handle features not explicitly selected
        if self.default is not False:
            # build JIT pipeline
            default_features = [(self._unselected_columns(X), self.default)]
            self.default_pipeline = make_feature_union(default_features)
            self.default_pipeline.fit(X, y)
        return self

    def transform(self, X):
        """
        Transform the given data. Assumes that fit has already been called.

        X       the data to transform
        """
        extracted = []
        if self.pipeline is not None:  # some columns selected
            extracted.append(handle_feature(self.pipeline.transform(X)))

        # handle features not explicitly selected
        if self.default is not False:
            Xt = self.default_pipeline.transform(X)
            extracted.append(handle_feature(Xt))

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
