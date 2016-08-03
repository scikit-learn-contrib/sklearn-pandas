import pytest
from sklearn_pandas.pipeline import TransformerPipeline, _call_fit

# In py3, mock is included with the unittest standard library
# In py2, it's a separate package
try:
    from unittest.mock import patch
except ImportError:
    from mock import patch


class NoTransformT(object):
    """Transformer without transform method.
    """
    def fit(self, x):
        return self


class NoFitT(object):
    """Transformer without fit method.
    """
    def transform(self, x):
        return self


class Trans(object):
    """
    Transformer with fit and transform methods
    """
    def fit(self, x, y=None):
        return self

    def transform(self, x):
        return self


def func_x_y(x, y, kwarg='kwarg'):
    """
    Function with required x and y arguments
    """
    return


def func_x(x, kwarg='kwarg'):
    """
    Function with required x argument
    """
    return


def func_raise_type_err(x, y, kwarg='kwarg'):
    """
    Function with required x and y arguments,
    raises TypeError
    """
    raise TypeError


def test_all_steps_fit_transform():
    """
    All steps must implement fit and transform. Otherwise, raise TypeError.
    """
    with pytest.raises(TypeError):
        TransformerPipeline([('svc', NoTransformT())])

    with pytest.raises(TypeError):
        TransformerPipeline([('svc', NoFitT())])


@patch.object(Trans, 'fit', side_effect=func_x_y)
def test_called_with_x_and_y(mock_fit):
    """
    Fit method with required X and y arguments is called with both and with
    any additional keywords
    """
    _call_fit(Trans().fit, 'X', 'y', kwarg='kwarg')
    mock_fit.assert_called_with('X', 'y', kwarg='kwarg')


@patch.object(Trans, 'fit', side_effect=func_x)
def test_called_with_x(mock_fit):
    """
    Fit method with a required X arguments is called with it and with
    any additional keywords
    """
    _call_fit(Trans().fit, 'X', 'y', kwarg='kwarg')
    mock_fit.assert_called_with('X', kwarg='kwarg')

    _call_fit(Trans().fit, 'X', kwarg='kwarg')
    mock_fit.assert_called_with('X', kwarg='kwarg')


@patch.object(Trans, 'fit', side_effect=func_raise_type_err)
def test_raises_type_error(mock_fit):
    """
    If a fit method with required X and y arguments raises a TypeError, it's
    re-raised (for a different reason) when it's called with one argument
    """
    with pytest.raises(TypeError):
        _call_fit(Trans().fit, 'X', 'y', kwarg='kwarg')
