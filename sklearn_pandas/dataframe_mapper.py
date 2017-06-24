import sys
import inspect
import pandas as pd
import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin

from .cross_validation import DataWrapper
from .pipeline import make_transformer_pipeline, _call_fit

# load in the correct stringtype: str for py3, basestring for py2
string_types = str if sys.version_info >= (3, 0) else basestring  # NOQA


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


def _build_feature(columns, transformers, options={}):
    return (columns, _build_transformer(transformers), options)


def _get_feature_names(estimator):
    """
    Attempt to extract feature names based on a given estimator.
    Note: this might break some of the existing pipeline as there might be
    actual names instead of _0, _1, ...
    """
    if hasattr(estimator, 'classes_'):
        return estimator.classes_
    elif hasattr(estimator, 'get_feature_names') \
            and inspect.ismethod(getattr(estimator, 'get_feature_names')):
        return estimator.get_feature_names()
    return None

class DataFrameMapper(BaseEstimator, TransformerMixin):
    """
    Map Pandas data frame column subsets to their own
    sklearn transformation.
    """

    def __init__(self, features, default=False, sparse=False, df_out=False,
                 input_df=False):
        """
        Params:

        features    a list of tuples with features definitions.
                    The first element is the pandas column selector. This can
                    be a string (for one column) or a list of strings.
                    The second element is an object that supports
                    sklearn's transform interface, or a list of such objects.
                    The third element is optional and, if present, must be
                    a dictionary with the options to apply to the
                    transformation. Example: {'alias': 'day_of_week'}

        default     default transformer to apply to the columns not
                    explicitly selected in the mapper. If False (default),
                    discard them. If None, pass them through untouched. Any
                    other transformer will be applied to all the unselected
                    columns as a whole, taken as a 2d-array.

        sparse      will return sparse matrix if set True and any of the
                    extracted features is sparse. Defaults to False.

        df_out      return a pandas data frame, with each column named using
                    the pandas column that created it (if there's only one
                    input and output) or the input columns joined with '_'
                    if there's multiple inputs, and the name concatenated with
                    '_1', '_2' etc if there's multiple outputs. NB: does not
                    work if *default* or *sparse* are true

        input_df    If ``True`` pass the selected columns to the transformers
                    as a pandas DataFrame or Series. Otherwise pass them as a
                    numpy array. Defaults to ``False``.
        """
        if isinstance(features, list):
            features = [_build_feature(*feature) for feature in features]
        self.features = features
        self.default = _build_transformer(default)
        self.sparse = sparse
        self.df_out = df_out
        self.input_df = input_df
        self.transformed_names_ = []

        if (df_out and (sparse or default)):
            raise ValueError("Can not use df_out with sparse or default")

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
        # compatibility for older versions of sklearn-pandas
        self.features = [_build_feature(*feat) for feat in state['features']]
        self.sparse = state.get('sparse', False)
        self.default = state.get('default', False)
        self.df_out = state.get('df_out', False)

        self.input_df = state.get('input_df', False)

    def _get_col_subset(self, X, cols, input_df=False):
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
            t = X[cols[0]]
        else:
            t = X[cols]

        # return either a DataFrame/Series or a numpy array
        if input_df:
            return t
        else:
            return t.values
        return t

    def fit(self, X, y=None):
        """
        Fit a transformation from the pipeline

        X       the data to fit

        y       the target vector relative to X, optional

        """

        for columns, transformers, options in self.features:
            input_df = options.get('input_df', self.input_df)

            if transformers is not None:
                _call_fit(transformers.fit,
                          self._get_col_subset(X, columns, input_df), y)

        # handle features not explicitly selected
        if self.default:  # not False and not None
            _call_fit(self.default.fit,
                      self._get_col_subset(
                          X, self._unselected_columns(X), self.input_df
                      ), y)
        return self

    def get_names(self, columns, transformer, x, alias=None):
        """
        Return verbose names for the transformed columns.

        columns       name (or list of names) of the original column(s)
        transformer   transformer
        x             transformed columns (numpy.ndarray)
        alias         base name to use for the selected columns
        """
        if alias is not None:
            name = alias
        elif isinstance(columns, list):
            name = '_'.join(columns)
        else:
            name = columns
        num_cols = x.shape[1] if len(x.shape) > 1 else 1
        if num_cols > 1:
            # If there are as many columns as classes in the transformer,
            # infer column names from classes names.
            names = _get_feature_names(transformer)
            if names is not None and len(names) == num_cols:
                return [name + '_' + str(o) for o in names]
            # otherwise, return name concatenated with '_1', '_2', etc.
            else:
                return [name + '_' + str(o) for o in range(num_cols)]
        else:
            return [name]

    def transform(self, X):
        """
        Transform the given data. Assumes that fit has already been called.

        X       the data to transform
        """
        extracted = []
        self.transformed_names_ = []
        for columns, transformers, options in self.features:
            input_df = options.get('input_df', self.input_df)
            # columns could be a string or list of
            # strings; we don't care because pandas
            # will handle either.
            Xt = self._get_col_subset(X, columns, input_df)
            if transformers is not None:
                Xt = transformers.transform(Xt)
            extracted.append(_handle_feature(Xt))

            alias = options.get('alias')
            self.transformed_names_ += self.get_names(
                columns, transformers, Xt, alias)

        # handle features not explicitly selected
        if self.default is not False:
            unsel_cols = self._unselected_columns(X)
            Xt = self._get_col_subset(X, unsel_cols, self.input_df)
            if self.default is not None:
                Xt = self.default.transform(Xt)
            extracted.append(_handle_feature(Xt))
            self.transformed_names_ += self.get_names(
                unsel_cols, self.default, Xt)

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

        if not self.df_out:
            return stacked

        return pd.DataFrame(stacked, columns=self.transformed_names_)
