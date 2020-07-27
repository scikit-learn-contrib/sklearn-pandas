import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin


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


class NumericalTransformer(TransformerMixin):
    """
    Provides commonly used numerical transformers.
    """
    SUPPORTED_FUNCTIONS = ['log', 'log1p']

    def __init__(self, func):
        """
        Params

        func    function to apply to input columns. The function will be
                applied to each value. Supported functions are defined
                in SUPPORTED_FUNCTIONS variable. Throws assertion error if the
                not supported.
        """
        assert func in self.SUPPORTED_FUNCTIONS, \
            f"Only following func are supported: {self.SUPPORTED_FUNCTIONS}"
        super(NumericalTransformer, self).__init__()
        self.__func = func

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if self.__func == 'log1p':
            return np.vectorize(np.log1p)(X)
        elif self.__func == 'log':
            return np.vectorize(np.log)(X)

        raise ValueError(f"Invalid function name: {self.__func}")
