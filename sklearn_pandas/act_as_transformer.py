from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np


class ActAsTransformer(BaseEstimator, TransformerMixin):
    """
    Use this class to convert a random function into a
    transformer.
    """

    def __init__(self, func):
        self.__func = func

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        return np.vectorize(self.__func)(x)

    def __call__(self, *args, **kwargs):
        return self.__func(*args, **kwargs)
