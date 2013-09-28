
__version__ = '0.0.6'

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn import cross_validation
from sklearn import grid_search

def cross_val_score(model, X, *args, **kwargs):
    X = DataWrapper(X)
    return cross_validation.cross_val_score(model, X, *args, **kwargs)


class GridSearchCV(grid_search.GridSearchCV):
    def fit(self, X, *params, **kwparams):
        super(GridSearchCV, self).fit(DataWrapper(X), *params, **kwparams)

    def predict(self, X, *params, **kwparams):
        super(GridSearchCV, self).fit(DataWrapper(X), *params, **kwparams)


class RandomizedSearchCV(grid_search.RandomizedSearchCV):
    def fit(self, X, *params, **kwparams):
        super(RandomizedSearchCV, self).fit(DataWrapper(X), *params, **kwparams)

    def predict(self, X, *params, **kwparams):
        super(RandomizedSearchCV, self).fit(DataWrapper(X), *params, **kwparams)


class DataWrapper(object):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, key):
        return self.df.iloc[key]
    

class DataFrameMapper(BaseEstimator, TransformerMixin):
    '''
    Map Pandas data frame column subsets to their own
    sklearn transformation.
    '''

    def __init__(self, features):
        '''
        Params:

        features    a list of pairs. The first element is the pandas column
                    selector. This can be a string (for one column) or a list
                    of strings. The second element is an object that supports
                    sklearn's transform interface.
        '''
        self.features = features


    def _get_col_subset(self, X, cols):
        '''
        Get a subset of columns from the given table X.

        X       a Pandas dataframe; the table to select columns from
        cols    a string or list of strings representing the columns
                to select

        Returns a numpy array with the data from the selected columns
        ''' 
        if isinstance(cols, basestring):
            cols = [cols]

        if isinstance(X, list):
            X = [x[cols] for x in X]
            X = pd.DataFrame(X)

        elif isinstance(X, DataWrapper):
            # if it's a datawrapper, unwrap it
            X = X.df

        if len(cols) == 1:
            t = X[cols[0]]
        else:
            t = X.as_matrix(cols)

        # there is an sklearn bug (#2374) which causes weird behavior
        # when 'object' type arrays are passed to labelling functions.
        # To get around this, in cases where all columns are strings
        # (represnted as object by Pandas), we convert the dtype to 
        # numpy's string type
        if np.all(X.dtypes[cols] == 'object'):
            t = np.array(t, dtype='|S')
         
        return t


    def fit(self, X, y=None):
        '''
        Fit a transformation from the pipeline

        X       the data to fit
        '''
        for columns, transformer in self.features:
            if transformer is not None:
                transformer.fit(self._get_col_subset(X, columns))
        return self


    def transform(self, X):
        '''
        Transform the given data. Assumes that fit has already been called.

        X       the data to transform
        '''
        extracted = []
        for columns, transformer in self.features:
            # columns could be a string or list of
            # strings; we don't care because pandas
            # will handle either.
            if transformer is not None:
                fea = transformer.transform(self._get_col_subset(X, columns))
            else:
                fea = self._get_col_subset(X, columns)
            
            if hasattr(fea, 'toarray'):
                # sparse arrays should be converted to regular arrays
                # for hstack.
                fea = fea.toarray()

            if len(fea.shape) == 1:
                fea = np.array([fea]).T
            extracted.append(fea)

        # combine the feature outputs into one array.
        # at this point we lose track of which features
        # were created from which input columns, so it's
        # assumed that that doesn't matter to the model.
        return np.hstack(extracted)

