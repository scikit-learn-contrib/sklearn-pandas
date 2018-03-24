import pandas as pd
import numpy as np


from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


def _get_mask(X, value):
    """
    Compute the boolean mask X == missing_values.
    """
    if value == "NaN" or \
       value is None or \
       (isinstance(value, float) and np.isnan(value)):
        return pd.isnull(X)
    else:
        return X == value


class CategoricalImputer(BaseEstimator, TransformerMixin):
    """
    Impute missing values from a categorical/string np.ndarray or pd.Series
    with the most frequent value on the training data.

    Parameters
    ----------
    missing_values : string or "NaN", optional (default="NaN")
        The placeholder for the missing values. All occurrences of
        `missing_values` will be imputed. None and np.nan are treated
        as being the same, use the string value "NaN" for them.

    copy : boolean, optional (default=True)
        If True, a copy of X will be created.

    replacement : string, optional (default="mode")
        The value that all occurrences of missing values are replaced
        with. Use this to specify an alternative if you don't want to
        impute with the mode, or if there are multiple modes in your
        data and you want to choose a particular one.

    Attributes
    ----------
    fill_ : str
        Most frequent value of the training data.

    """

    def __init__(self, missing_values='NaN', replacement='mode', copy=True):
        self.missing_values = missing_values
        self.copy = copy
        self.replacement = replacement

    def fit(self, X, y=None):
        """

        Get the most frequent value.

        Parameters
        ----------
            X : np.ndarray or pd.Series
                Training data.

            y : Passthrough for ``Pipeline`` compatibility.

        Returns
        -------
            self: CategoricalImputer
        """

        mask = _get_mask(X, self.missing_values)
        X = X[~mask]
        if self.replacement == 'mode':
            modes = pd.Series(X).mode()
        else:
            modes = np.array([self.replacement])
        if modes.shape[0] == 0:
            raise ValueError('No value is repeated more than '
                             'once in the column')
        else:
            self.fill_ = modes[0]

        return self

    def transform(self, X):
        """

        Replaces missing values in the input data with the most frequent value
        of the training data.

        Parameters
        ----------
            X : np.ndarray or pd.Series
                Data with values to be imputed.

        Returns
        -------
            np.ndarray
                Data with imputed values.
        """

        check_is_fitted(self, 'fill_')

        if self.copy:
            X = X.copy()

        mask = _get_mask(X, self.missing_values)
        X[mask] = self.fill_

        return np.asarray(X)
