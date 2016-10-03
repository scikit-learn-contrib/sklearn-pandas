'''
an alternative implementation which uses just sklearn Pipeline and FeatureUnion.
This makes the resultant transformer more compatible with other scikit-learn APIs.
'''
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
from sklearn.pipeline import FeatureUnion
from .pipeline import TransformerPipeline


def _handle_feature(fea):
    """
    Convert 1-dimensional arrays to 2-dimensional column vectors.
    """
    if len(fea.shape) == 1:
        fea = np.array([fea]).T

    return fea







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
