import pytest
from pipeline import TransformerPipeline


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


def test_all_steps_fit_transform():
    """
    All steps must implement fit and transform. Otherwise, raise TypeError.
    """
    with pytest.raises(TypeError):
        TransformerPipeline([('svc', NoTransformT())])

    with pytest.raises(TypeError):
        TransformerPipeline([('svc', NoFitT())])
