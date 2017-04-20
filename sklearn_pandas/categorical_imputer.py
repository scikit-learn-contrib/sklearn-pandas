"""

Impute missing values from a categorical/string np.ndarray or pd.Series with
the most frequent value on the training data.

"""

import pandas as pd
import numpy as np

from sklearn.base import TransformerMixin


class CategoricalImputer(TransformerMixin):

    """

    Attributes
    ----------

    fill : str
        Most frequent value of the training data.

    """

    def __init__(self):

        self.fill = None

    def fit(self, X):

        """

        Get the most frequent value.

        Parameters
        ----------
            X : np.ndarray or pd.Series
                Training data.

        Returns
        -------
        CategoricalImputer
            Itself.

        """

        self.fill = pd.Series(X).mode().values[0]

        return self

    def transform(self, X):

        """

        Replaces null values in the input data with the most frequent value
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

        X = X.copy()

        X[pd.isnull(X)] = self.fill

        return np.asarray(X)
