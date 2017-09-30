import pytest

from sklearn_pandas.features_generator import gen_features


class MockClass(object):

    def __init__(self, value=1, name='class'):
        self.value = value
        self.name = name


@pytest.mark.parametrize('columns', [['colA', 'colB', 'colC']])
def test_generate_features_with_default_parameters(columns):
    """
    Tests generating features from classes with default init arguments
    """
    feature_defs = gen_features(columns=columns, classes=[MockClass])
    assert len(feature_defs) == len(columns)

    feature_dict = dict(feature_defs)
    assert columns == sorted(feature_dict)

    expected = {'value': 1, 'name': 'class'}
    for column, transformers in feature_dict.items():
        for obj in transformers:
            assert_attributes(obj, **expected)


def test_generate_features_with_several_classes():
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


def test_generate_features_with_none_transformers():
    feature_defs = gen_features(
        columns=['colA', 'colB', 'colC'], classes=[None])

    expected = [('colA', None),
                ('colB', None),
                ('colC', None)]

    assert feature_defs == expected


def assert_attributes(obj, **attrs):
    for attr, value in attrs.items():
        assert getattr(obj, attr) == value
