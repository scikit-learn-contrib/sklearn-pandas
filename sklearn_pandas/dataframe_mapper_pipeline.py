'''
an alternative implementation which uses just sklearn Pipeline and FeatureUnion.
This makes the resultant transformer more compatible with other scikit-learn APIs.
'''
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
from sklearn.pipeline import Pipeline, FeatureUnion


class ColumnSelectTransformer(BaseEstimator, TransformerMixin):
    '''
    A simple Transformer which selects a column or a group of columns from a Pandas' DataFrame
    '''

    def __init__(self, column_name):
        '''
        A Transformer which selects a column or a group of columns from a Pandas' DataFrame
        :param column_name: string or list of strings of columns to select
        '''
        self.column_name = column_name

    def fit(self, X, y=None):
        if not (isinstance(X, pd.DataFrame) or isinstance(X, pd.Series)):
            raise TypeError('Input should be a Pandas DataFrame or a Series (was %s)' % type(X))
        column_name = self.column_name
        if not isinstance(column_name, list):  # in case in bracketed as [] to output a (n,1) rather (n,) shape
            column_name = [column_name]
        for name in column_name:
            if name not in X.columns:
                raise ValueError('Select column name %s is not in %s' % (name, X.columns))
        return self

    def transform(self, X, y=None):
        return X[self.column_name]


def mapping_to_pipeline(mapping, n_jobs=-1):
    '''
    creates a pipeline from a mapping object by prefixing with ColumnSelectTransformer
    :param mapping: a list of tuples where the first is the column name(s) and the other is the transormation or list of transformation to apply. See DataFrameMapper for more information
    :param n_jobs: whether to calculate
    :return:
    '''
    union_list = []  # list of pipelines to union
    for column_name, transformer in mapping:  # loop over the mapping list
        if not isinstance(transformer, list):
            transformer = [transformer]
        column_name_str = column_name
        if not isinstance(column_name_str, str):
            column_name_str = str(column_name_str)
        selector = [('%s_selector' % column_name_str, ColumnSelectTransformer(column_name))]
        # turn extractor into a list for pipelining
        extractor = [('%s_extractor_stage_%d' % (column_name_str, idx), m) for idx, m in enumerate(transformer)]
        pipeline_list = selector
        if extractor is not None:
            pipeline_list += extractor
        pipe = Pipeline(pipeline_list)  # pipe line of selector followed by transformer
        union_list.append((column_name_str, pipe))  # add to the pipeline list
    pipe_union = FeatureUnion(union_list, n_jobs=n_jobs)  # merge pipelines into a concatenated form
    return pipe_union


import unittest


class TestPipelineMapping(unittest.TestCase):
    def setUp(self):
        from sklearn.datasets import load_boston
        data = load_boston()
        fm = data['data']
        y = data['target']
        columns = data['feature_names']
        df = pd.DataFrame(fm, columns=columns)
        self.df = df
        self.y = y
        from sklearn.preprocessing import StandardScaler
        from sklearn.preprocessing import OneHotEncoder
        self.mapping = [(['AGE'], StandardScaler()),
                        (['RAD'], OneHotEncoder(handle_unknown="ignore"))
                        ]

    def test_make_pipe(self):
        try:
            pipeline = mapping_to_pipeline(mapping=self.mapping)
        except Exception as e:
            self.fail('Unexpected exception raised:', e)
        self.assertTrue(isinstance(pipeline, FeatureUnion))

    def test_transform(self):
        pipeline = mapping_to_pipeline(mapping=self.mapping)
        n_unique = self.df.apply(lambda x: x.nunique())
        try:
            transformed = pipeline.fit_transform(self.df)
        except Exception as e:
            self.fail('Unexpected exception raised:', e)
        self.assertEqual(self.df.shape[0], transformed.shape[0])
        self.assertEqual(n_unique['RAD'] + 1, transformed.shape[1])

    def test_pipe_cv(self):
        pipeline = mapping_to_pipeline(mapping=self.mapping)
        from sklearn.linear_model import LinearRegression
        from sklearn.pipeline import Pipeline, make_pipeline
        full_pipeline = make_pipeline(pipeline, LinearRegression())
        from sklearn.cross_validation import cross_val_score
        try:
            scores = cross_val_score(full_pipeline, self.df, self.y)
        except Exception as e:
            self.fail('Unexpected exception raised:', e)


if __name__ == '__main__':
    unittest.main()
