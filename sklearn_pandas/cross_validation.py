from sklearn import cross_validation
from sklearn import grid_search


def cross_val_score(model, X, *args, **kwargs):
    X = DataWrapper(X)
    return cross_validation.cross_val_score(model, X, *args, **kwargs)


class GridSearchCV(grid_search.GridSearchCV):
    def fit(self, X, *params, **kwparams):
        return super(GridSearchCV, self).fit(DataWrapper(X), *params, **kwparams)

    def predict(self, X, *params, **kwparams):
        return super(GridSearchCV, self).predict(DataWrapper(X), *params, **kwparams)


try:
    class RandomizedSearchCV(grid_search.RandomizedSearchCV):
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
