"""

Impute missing values.

"""

import pandas as pd
import numpy as np

from sklearn.base import TransformerMixin


class DataFrameImputer(TransformerMixin):

    """

    Impute missing values.

    Columns of dtype object are imputed with the most frequent value in column.

    Columns of other types are imputed with mean of column.

    """

    def __init__(self):

        self.fill = None

    def fit(self, X):

        """

        Get a Series with the imputer value for each of the columns in X.

        :returns: DataFrameImputer

        """

        imputer_values = []
        for col in X:
            if X[col].dtype == np.dtype('O'):
                imputer_values += [X[col].value_counts().index[0]]
            else:
                imputer_values += [X[col].mean()]

        self.fill = pd.Series(imputer_values, index=X.columns)

        return self

    def transform(self, X):

        """

        Replaces null values in the dataframe with the imputer value for each of the columns in X.

        :returns: DataFrame

        """

        return X.fillna(self.fill)
