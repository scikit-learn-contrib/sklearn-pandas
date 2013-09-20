import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn import cross_validation

def cross_val_score(model, X, *args, **kwargs):
    class ModelDataWrapper(BaseEstimator):
        '''
        sklearn's built-in cross-validation will turn a Pandas DataFrame
        into an array, which clobbers all the  column names. This
        breaks DataFrameMapper (which uses the column names), so we need
        to work around it.

        We do this by wrapping the model and data in this wrapper class.
        We don't pass the data to sklearn's cross_val_score at all, we
        just pass a list of indices to the data. Then sklearn picks
        a subset of those indices and passes them to this class (as
        the argument to fit, predict, or transform). This class selects
        the appropriate subset of rows from the original dataset,
        and sends it off to the model without stripping the column names.
        '''

        def __init__(self, model, X):
            '''
            Create a ModelDataWrapper

            model   the model to cross-validate
            X       the data to use for cross-validation
            '''
            self.model = model
            self.X = X

        def fit(self, x, y):
            self.model.fit(self._get_row_subset(x), y)
            return self

        def predict(self, x):
            return self.model.predict(self._get_row_subset(x))

        def _get_row_subset(self, rows):
            '''
            Return a dataframe with rows matching the indices
            provided

            rows    a list of indices of rows to return
            '''
            return self.X.iloc[rows].reset_index(drop=True)
    

    X_indices = xrange(len(X))
    mdw = ModelDataWrapper(model, X)
    return cross_validation.cross_val_score(mdw, X_indices, *args, **kwargs)


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
            fea = transformer.transform(self._get_col_subset(X, columns))
            
            if len(fea.shape) == 1:
                fea = np.array([fea]).T
            extracted.append(fea)

        # combine the feature outputs into one array.
        # at this point we lose track of which features
        # were created from which input columns, so it's
        # assumed that that doesn't matter to the model.
        return np.hstack(extracted)

