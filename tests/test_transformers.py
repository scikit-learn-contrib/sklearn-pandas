import tempfile
import pytest
import numpy as np
from pandas import DataFrame
import joblib

from sklearn_pandas import DataFrameMapper
from sklearn_pandas import NumericalTransformer


@pytest.fixture
def simple_dataset():
    return DataFrame({
        'feat1': [1, 2, 1, 3, 1],
        'feat2': [1, 2, 2, 2, 3],
        'feat3': [1, 2, 3, 4, 5],
    })


def test_common_numerical_transformer(simple_dataset):
    """
    Test log transformation
    """
    transfomer = DataFrameMapper([
        ('feat1', NumericalTransformer('log')),
        ('feat2', NumericalTransformer('sin')),
        ('feat3', NumericalTransformer('cos'))
    ], df_out=True)
    df = simple_dataset
    outDF = transfomer.fit_transform(df)
    assert list(outDF.columns) == ['feat1', 'feat2', 'feat3']
    assert np.array_equal(df['feat1'].apply(np.log).values, outDF.feat1.values)
    assert np.array_equal(df['feat2'].apply(np.sin).values, outDF.feat2.values)
    assert np.array_equal(df['feat3'].apply(np.cos).values, outDF.feat3.values)


def test_numerical_transformer_serialization(simple_dataset):
    """
    Test if you can serialize transformer
    """
    transfomer = DataFrameMapper([
        ('feat1', NumericalTransformer('log')),
        ('feat2', NumericalTransformer('sin')),
        ('feat3', NumericalTransformer('cos')),
    ])

    df = simple_dataset
    transfomer.fit(df)
    f = tempfile.NamedTemporaryFile(delete=True)
    joblib.dump(transfomer, f.name)
    transfomer2 = joblib.load(f.name)
    np.array_equal(transfomer.transform(df), transfomer2.transform(df))
    f.close()
