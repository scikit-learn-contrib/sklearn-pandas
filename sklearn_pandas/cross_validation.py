import warnings
from sklearn import cross_validation
from sklearn import grid_search

DEPRECATION_MSG = '''
    Custom cross-validation compatibility shims are no longer needed for
    scikit-learn>=0.16.0 and will be dropped in sklearn-pandas==2.0.
'''


def cross_val_score(model, X, *args, **kwargs):
    warnings.warn(DEPRECATION_MSG, DeprecationWarning)
    X = DataWrapper(X)
    return cross_validation.cross_val_score(model, X, *args, **kwargs)


class GridSearchCV(grid_search.GridSearchCV):
    def __init__(self, *args, **kwargs):
        warnings.warn(DEPRECATION_MSG, DeprecationWarning)
        super(GridSearchCV, self).__init__(*args, **kwargs)

    def fit(self, X, *params, **kwparams):
        return super(GridSearchCV, self).fit(DataWrapper(X), *params, **kwparams)

    def predict(self, X, *params, **kwparams):
        return super(GridSearchCV, self).predict(DataWrapper(X), *params, **kwparams)


try:
    class RandomizedSearchCV(grid_search.RandomizedSearchCV):
        def __init__(self, *args, **kwargs):
            warnings.warn(DEPRECATION_MSG, DeprecationWarning)
            super(RandomizedSearchCV, self).__init__(*args, **kwargs)

        def fit(self, X, *params, **kwparams):
            return super(RandomizedSearchCV, self).fit(DataWrapper(X), *params, **kwparams)

        def predict(self, X, *params, **kwparams):
            return super(RandomizedSearchCV, self).predict(DataWrapper(X), *params, **kwparams)
except AttributeError:
    pass


class DataWrapper(object):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, key):
        return self.df.iloc[key]
