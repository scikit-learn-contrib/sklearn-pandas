from collections import Counter

import pytest
import numpy as np
from pandas import DataFrame
from numpy.testing import assert_array_equal

from sklearn_pandas import DataFrameMapper
from sklearn_pandas.features_generator import gen_features


class MockClass(object):

    def __init__(self, value=1, name='class'):
        self.value = value
        self.name = name


class MockTransformer(object):

    def __init__(self):
        self.most_common_ = None

    def fit(self, X, y=None):
        [(value, _)] = Counter(X).most_common(1)
        self.most_common_ = value
        return self

    def transform(self, X, y=None):
        return np.asarray([self.most_common_] * len(X))


@pytest.fixture
def simple_dataset():
    return DataFrame({
        'feat1': [1, 2, 1, 3, 1],
        'feat2': [1, 2, 2, 2, 3],
        'feat3': [1, 2, 3, 4, 5],
    })


def test_generate_features_with_default_parameters():
    """
    Tests generating features from classes with default init arguments.
    """
    columns = ['colA', 'colB', 'colC']
    feature_defs = gen_features(columns=columns, classes=[MockClass])
    assert len(feature_defs) == len(columns)

    feature_dict = dict(feature_defs)
    assert columns == sorted(feature_dict.keys())

    # default init arguments for MockClass for clarification.
    expected = {'value': 1, 'name': 'class'}
    for column, transformers in feature_dict.items():
        for obj in transformers:
            assert_attributes(obj, **expected)


def test_generate_features_with_several_classes():
    """
    Tests generating features pipeline with different transformers parameters.
    """
    feature_defs = gen_features(
        columns=['colA', 'colB', 'colC'],
        classes=[
            {'class': MockClass},
            {'class': MockClass, 'name': 'mockA'},
            {'class': MockClass, 'name': 'mockB', 'value': None}
        ]
    )

    for transformers in dict(feature_defs).values():
        assert_attributes(transformers[0], name='class', value=1)
        assert_attributes(transformers[1], name='mockA', value=1)
        assert_attributes(transformers[2], name='mockB', value=None)


def test_generate_features_with_none_only_transformers():
    """
    Tests generating "dummy" feature definition which doesn't apply any
    transformation.
    """
    feature_defs = gen_features(
        columns=['colA', 'colB', 'colC'], classes=[None])

    expected = [('colA', None),
                ('colB', None),
                ('colC', None)]

    assert feature_defs == expected


def test_compatibility_with_data_frame_mapper(simple_dataset):
    """
    Tests compatibility of generated feature definition with DataFrameMapper.
    """
    features_defs = gen_features(
        columns=['feat1', 'feat2'],
        classes=[MockTransformer])
    features_defs.append(('feat3', None))

    mapper = DataFrameMapper(features_defs)
    X = mapper.fit_transform(simple_dataset)
    expected = np.asarray([
        [1, 2, 1],
        [1, 2, 2],
        [1, 2, 3],
        [1, 2, 4],
        [1, 2, 5]
    ])

    assert_array_equal(X, expected)


def assert_attributes(obj, **attrs):
    for attr, value in attrs.items():
        assert getattr(obj, attr) == value
